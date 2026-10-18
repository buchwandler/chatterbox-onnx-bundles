#!/usr/bin/env python3
"""Check the generated Chatterbox bundle catalog without installing a package."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "catalog" / "bundles.json"
DEFAULT_SOURCE = ROOT / "catalog" / "source.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "bundle-catalog.schema.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
EXPECTED = {
    "nano-v2": {
        "repository": "KitsuMate/chatterbox-nano-v2-onnx",
        "profiles": {
            "cpu": {
                "speech_encoder": "fp32",
                "embed_tokens": "fp32",
                "language_model": "int8_mixed",
                "conditional_decoder": "int8",
            },
            "fp32": dict.fromkeys(COMPONENTS, "fp32"),
        },
        "default_profile": "cpu",
        "tokenizer_contract": "nano-v2",
        "static_roles": {"tokenizer_config", "upstream_manifest"},
        "static_paths": {"tokenizer_config": "tokenizer_config.json", "upstream_manifest": "manifest.json"},
        "external_data": {
            ("conditional_decoder", "fp32", "onnx/conditional_decoder.onnx_data"),
            ("conditional_decoder", "int8", "onnx/conditional_decoder_int8.onnx.data"),
            ("language_model", "int8_mixed", "onnx/language_model_int8_mixed.onnx.data"),
        },
    },
    "turbo": {
        "repository": "ResembleAI/chatterbox-turbo-ONNX",
        "profiles": {
            "fp32": dict.fromkeys(COMPONENTS, "fp32"),
            "q4f16": dict.fromkeys(COMPONENTS, "q4f16"),
        },
        "default_profile": "fp32",
        "tokenizer_contract": "turbo",
        "static_roles": {"model_config", "generation_config"},
        "static_paths": {"model_config": "config.json", "generation_config": "generation_config.json"},
        "external_data": {
            (component, quality, f"onnx/{component}{suffix}.onnx_data")
            for component in COMPONENTS
            for quality, suffix in (("fp32", ""), ("q4f16", "_q4f16"))
        },
    },
}
STATIC_ROLES = {"tokenizer", "tokenizer_config", "upstream_manifest", "model_config", "generation_config"}
ARTIFACT_ROLES = STATIC_ROLES | {"graph", "external_data"}


def _read_json(path: Path, label: str, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot read valid JSON ({exc})")
        return None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in value.split("/"))


def _check_artifact(
    artifact: Any,
    *,
    label: str,
    repository: str,
    revision: str,
    errors: list[str],
) -> None:
    if not isinstance(artifact, dict):
        errors.append(f"{label}: artifact must be an object")
        return
    required = {
        "component", "filename", "quality", "role", "sha256", "size", "source_path", "url"
    }
    if set(artifact) != required:
        errors.append(f"{label}: artifact fields differ from the catalog schema")
        return
    role = artifact["role"]
    component = artifact["component"]
    quality = artifact["quality"]
    if not isinstance(role, str) or role not in ARTIFACT_ROLES:
        errors.append(f"{label}: unknown artifact role {role!r}")
        return
    if role in {"graph", "external_data"}:
        if component not in COMPONENTS or not isinstance(quality, str) or not quality:
            errors.append(f"{label}: graph artifacts require a known component and quality")
    elif component is not None or quality is not None:
        errors.append(f"{label}: static artifacts cannot have component or quality selectors")
    if (
        not isinstance(artifact["filename"], str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", artifact["filename"]) is None
    ):
        errors.append(f"{label}: filename must be a safe basename")
    source_path = artifact["source_path"]
    if not _safe_relative_path(source_path):
        errors.append(f"{label}: source_path must be a safe relative path")
    elif artifact["filename"] != PurePosixPath(source_path).name:
        errors.append(f"{label}: filename does not match source_path")
    if not _positive_int(artifact["size"]):
        errors.append(f"{label}: size must be a positive integer")
    if not isinstance(artifact["sha256"], str) or SHA256_RE.fullmatch(artifact["sha256"]) is None:
        errors.append(f"{label}: sha256 must be 64 lowercase hexadecimal characters")
    url = artifact["url"]
    if not isinstance(url, str):
        errors.append(f"{label}: url must be a string")
        return
    parsed = urlsplit(url)
    expected_prefix = f"/{repository}/resolve/{revision}/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or not parsed.path.startswith(expected_prefix)
        or parsed.path[len(expected_prefix) :] != source_path
        or parsed.query not in {"", "download=true"}
        or parsed.fragment
    ):
        errors.append(f"{label}: URL is not pinned to the artifact source path and revision")
    else:
        path_revision = parsed.path[len(f"/{repository}/resolve/") :].split("/", 1)[0]
        if REVISION_RE.fullmatch(path_revision) is None or path_revision != revision:
            errors.append(f"{label}: URL revision is not an exact pinned commit")


def _check_bundle(bundle_id: str, bundle: Any, aliases: set[str], errors: list[str]) -> int:
    expected = EXPECTED[bundle_id]
    if not isinstance(bundle, dict):
        errors.append(f"{bundle_id}: bundle must be an object")
        return 0
    required_fields = {
        "aliases", "artifacts", "default_profile", "frontend", "id", "language",
        "profiles", "runtime", "sample_rate", "source", "system",
    }
    if set(bundle) != required_fields:
        errors.append(f"{bundle_id}: bundle fields differ from the catalog schema")
    if bundle.get("id") != bundle_id or bundle.get("system") != "chatterbox":
        errors.append(f"{bundle_id}: id or system does not match its catalog key")
    if bundle.get("language") != "en" or bundle.get("sample_rate") != 24000:
        errors.append(f"{bundle_id}: expected English and native 24000 Hz audio")
    if bundle.get("default_profile") != expected["default_profile"]:
        errors.append(f"{bundle_id}: default profile is not the supported MVP default")

    bundle_aliases = bundle.get("aliases")
    if not isinstance(bundle_aliases, list) or any(
        not isinstance(alias, str) or SAFE_NAME_RE.fullmatch(alias) is None for alias in bundle_aliases
    ):
        errors.append(f"{bundle_id}: aliases must be a list of non-empty strings")
        bundle_aliases = []
    for alias in [bundle_id, *bundle_aliases]:
        if alias.casefold() in aliases:
            errors.append(f"{bundle_id}: alias {alias!r} collides with another bundle")
        aliases.add(alias.casefold())

    frontend = bundle.get("frontend")
    if not isinstance(frontend, dict) or frontend != {
        "tokenizer_contract": expected["tokenizer_contract"],
        "tokenizer_path": "tokenizer.json",
        "tokenizer_role": "tokenizer",
    }:
        errors.append(f"{bundle_id}: frontend metadata does not match its export contract")

    runtime = bundle.get("runtime")
    required_runtime = {
        "start_speech_token", "stop_speech_token", "silence_token", "trailing_silence_tokens",
        "default_max_new_tokens", "default_repetition_penalty",
    }
    if not isinstance(runtime, dict) or set(runtime) != required_runtime:
        errors.append(f"{bundle_id}: runtime metadata fields differ from the normalized contract")
    else:
        for name in ("start_speech_token", "stop_speech_token", "silence_token"):
            if not isinstance(runtime[name], int) or isinstance(runtime[name], bool) or runtime[name] < 0:
                errors.append(f"{bundle_id}: runtime {name} must be a non-negative integer")
        for name in ("trailing_silence_tokens",):
            if not isinstance(runtime[name], int) or isinstance(runtime[name], bool) or runtime[name] < 0:
                errors.append(f"{bundle_id}: runtime {name} must be a non-negative integer")
        if not _positive_int(runtime["default_max_new_tokens"]):
            errors.append(f"{bundle_id}: default_max_new_tokens must be positive")
        penalty = runtime["default_repetition_penalty"]
        if (
            isinstance(penalty, bool)
            or not isinstance(penalty, (int, float))
            or (isinstance(penalty, float) and not math.isfinite(penalty))
            or penalty <= 0
        ):
            errors.append(f"{bundle_id}: default_repetition_penalty must be positive")

    source = bundle.get("source")
    if not isinstance(source, dict):
        errors.append(f"{bundle_id}: source must be an object")
        source = {}
    if set(source) != {"provider", "repository", "requested_revision", "revision"}:
        errors.append(f"{bundle_id}: source fields differ from the catalog schema")
    repository = expected["repository"]
    revision = source.get("revision")
    if (
        source.get("provider") != "huggingface"
        or source.get("repository") != repository
        or not isinstance(source.get("requested_revision"), str)
        or not source["requested_revision"]
        or not isinstance(revision, str)
        or REVISION_RE.fullmatch(revision) is None
    ):
        errors.append(f"{bundle_id}: source identity or resolved revision is invalid")
        revision = "0" * 40

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{bundle_id}: artifacts must be a non-empty array")
        artifacts = []
    for index, artifact in enumerate(artifacts):
        _check_artifact(
            artifact,
            label=f"{bundle_id}.artifacts[{index}]",
            repository=repository,
            revision=revision,
            errors=errors,
        )

    profiles = bundle.get("profiles")
    if not isinstance(profiles, dict) or profiles != expected["profiles"]:
        errors.append(f"{bundle_id}: profile map differs from the supported MVP graph selection")
        profiles = {}
    graph_keys: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("role") != "graph":
            continue
        component = artifact.get("component")
        quality = artifact.get("quality")
        if not isinstance(component, str) or not isinstance(quality, str):
            continue
        key = (component, quality)
        if key in graph_keys:
            errors.append(f"{bundle_id}: duplicate graph for component/quality {key!r}")
        graph_keys.add(key)
    for profile_name, components in expected["profiles"].items():
        if profiles.get(profile_name) != components:
            continue
        for component, quality in components.items():
            matches = [
                artifact for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("role") == "graph"
                and artifact.get("component") == component
                and artifact.get("quality") == quality
            ]
            if len(matches) != 1:
                errors.append(
                    f"{bundle_id}.{profile_name}: expected one {component} graph at quality {quality}, found {len(matches)}"
                )

    expected_graph_keys = {
        (component, quality)
        for profile in expected["profiles"].values()
        for component, quality in profile.items()
    }
    if graph_keys != expected_graph_keys:
        errors.append(f"{bundle_id}: graph inventory contains missing or unsupported profile artifacts")
    external_keys = {
        (item.get("component"), item.get("quality"), item.get("source_path"))
        for item in artifacts
        if isinstance(item, dict)
        and item.get("role") == "external_data"
        and isinstance(item.get("component"), str)
        and isinstance(item.get("quality"), str)
        and isinstance(item.get("source_path"), str)
    }
    if external_keys != expected["external_data"]:
        errors.append(f"{bundle_id}: external-data inventory differs from expected graph companions")
    external_data = [item for item in artifacts if isinstance(item, dict) and item.get("role") == "external_data"]
    for sidecar in external_data:
        matches = [
            graph for graph in artifacts
            if isinstance(graph, dict)
            and graph.get("role") == "graph"
            and graph.get("component") == sidecar.get("component")
            and graph.get("quality") == sidecar.get("quality")
            and isinstance(graph.get("source_path"), str)
            and isinstance(sidecar.get("source_path"), str)
            and sidecar.get("source_path") in {
                graph.get("source_path", "") + ".data",
                graph.get("source_path", "") + "_data",
            }
        ]
        if len(matches) != 1:
            errors.append(f"{bundle_id}: external-data artifact has no unique matching graph companion")

    static_artifacts = [item for item in artifacts if isinstance(item, dict) and isinstance(item.get("role"), str) and item.get("role") in STATIC_ROLES]
    required_roles = {"tokenizer", *expected["static_roles"]}
    actual_static_roles = {item["role"] for item in static_artifacts}
    if actual_static_roles != required_roles:
        errors.append(f"{bundle_id}: static artifact roles differ from the expected inventory")
    for role, source_path in expected["static_paths"].items():
        matches = [item for item in static_artifacts if item.get("role") == role]
        if len(matches) == 1 and matches[0].get("source_path") != source_path:
            errors.append(f"{bundle_id}: static artifact {role} has an unexpected source path")
    for role in required_roles:
        matches = [item for item in static_artifacts if item.get("role") == role]
        if len(matches) != 1:
            errors.append(f"{bundle_id}: expected exactly one static artifact with role {role}")
    tokenizer = [item for item in static_artifacts if item.get("role") == "tokenizer"]
    if (
        len(tokenizer) == 1
        and isinstance(frontend, dict)
        and tokenizer[0].get("source_path") != frontend.get("tokenizer_path")
    ):
        errors.append(f"{bundle_id}: tokenizer artifact path does not match frontend metadata")
    return len(artifacts)


def _check_source_snapshot(catalog: Any, source: Any, artifact_count: int, errors: list[str]) -> None:
    bundles = catalog.get("bundles") if isinstance(catalog, dict) else None
    if not isinstance(source, dict) or not isinstance(bundles, dict):
        errors.append("source snapshot or catalog bundles are not objects")
        return
    if set(source) != {"schema", "kind", "bundle_count", "artifact_count", "sources"}:
        errors.append("source snapshot fields differ from the catalog schema")
    if source.get("schema") != 1 or source.get("kind") != "chatterbox-onnx-source-snapshot":
        errors.append("source snapshot schema or kind is invalid")
    if source.get("bundle_count") != len(bundles) or source.get("artifact_count") != artifact_count:
        errors.append("source snapshot counts do not match the catalog")
    source_rows = source.get("sources")
    if not isinstance(source_rows, list):
        errors.append("source snapshot sources must be an array")
        return
    by_id: dict[str, dict[str, Any]] = {}
    for item in source_rows:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            errors.append("source snapshot contains a malformed source row")
            return
        source_id = item["id"]
        if source_id in by_id:
            errors.append(f"source snapshot contains duplicate bundle ID {source_id!r}")
        by_id[source_id] = item
    if set(by_id) != set(bundles):
        errors.append("source snapshot entries do not exactly match catalog bundle IDs")
        return
    for bundle_id, bundle in bundles.items():
        source_row = by_id[bundle_id]
        bundle_source = bundle.get("source") if isinstance(bundle, dict) else None
        if not isinstance(bundle_source, dict) or source_row != {"id": bundle_id, **bundle_source}:
            errors.append(f"{bundle_id}: source snapshot row differs from bundle provenance")


def _check_data_only_layout(errors: list[str]) -> None:
    forbidden = ("pyproject.toml", "setup.py")
    for name in forbidden:
        if (ROOT / name).exists():
            errors.append(f"data-only repository must not contain {name}")
    if (ROOT / "src").exists():
        errors.append("data-only repository must not contain a src package tree")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or any(part.startswith(".") and part.endswith("_cache") for part in path.parts):
            continue
        if path.is_file() and path.suffix.casefold() in {
            ".onnx", ".onnx_data", ".data", ".safetensors", ".pt", ".pth", ".bin"
        }:
            errors.append(f"data-only repository contains a model/weight file: {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    errors: list[str] = []
    catalog = _read_json(args.catalog, "bundle catalog", errors)
    source = _read_json(args.source, "source snapshot", errors)
    schema = _read_json(args.schema, "catalog schema", errors)
    if not isinstance(schema, dict):
        errors.append("catalog schema must be a JSON object")
    _check_data_only_layout(errors)

    if not isinstance(catalog, dict):
        errors.append("bundle catalog must be a JSON object")
        bundle_entries: dict[str, Any] = {}
    else:
        if set(catalog) != {"schema", "kind", "bundles"}:
            errors.append("bundle catalog fields differ from the catalog schema")
        if catalog.get("schema") != 1 or catalog.get("kind") != "chatterbox-onnx-bundle-catalog":
            errors.append("bundle catalog schema or kind is invalid")
        raw_bundles = catalog.get("bundles")
        if not isinstance(raw_bundles, dict):
            errors.append("bundle catalog bundles must be an object")
            bundle_entries = {}
        else:
            bundle_entries = raw_bundles
    if set(bundle_entries) != set(EXPECTED):
        errors.append("bundle catalog must contain exactly nano-v2 and turbo")

    aliases: set[str] = set()
    total_artifacts = 0
    for bundle_id, bundle in bundle_entries.items():
        if bundle_id in EXPECTED:
            total_artifacts += _check_bundle(bundle_id, bundle, aliases, errors)
    _check_source_snapshot(catalog, source, total_artifacts, errors)

    if errors:
        print("Chatterbox catalog integrity check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Chatterbox catalog integrity passed: bundles={len(bundle_entries)} "
        f"artifacts={total_artifacts}; no model weights or package metadata found"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
