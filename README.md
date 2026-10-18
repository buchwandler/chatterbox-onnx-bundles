# chatterbox-onnx-bundles

A reproducible, data-only catalog for the pinned Chatterbox ONNX exports used by
[OnnxVoice](https://github.com/buchwandler/onnxvoice) and
[ChatterboxSynth](https://github.com/buchwandler/chatterboxsynth). This repository records
where artifacts live and how to verify them; it is not a model mirror.

## Scope and ownership

This repository contains only catalog metadata, source provenance, a JSON Schema, integrity
checks, documentation, and CI configuration. It is not a Python package, has no installable
runtime or CLI, and does not contain ONNX weights or reference recordings.

OnnxVoice owns catalog interpretation, downloads, checksums, installation/cache management,
ONNX sessions, and the graph adapter. ChatterboxSynth owns the public synthesis API and
model-independent text/audio frontends. This repository owns the upstream artifact inventory
and profile metadata only.

## Supported bundles

| Bundle | Aliases | Profiles | Default | Native audio |
|---|---|---|---|---:|
| `nano-v2` | `nano`, `nano-110m` | `cpu`, `fp32` | `cpu` | 24 kHz |
| `turbo` | `turbo-350m` | `fp32`, `q4f16` | `fp32` | 24 kHz |

Each profile selects exactly four components: `speech_encoder`, `embed_tokens`, `language_model`,
and `conditional_decoder`. Nano `cpu` uses the pinned FP32 speech encoder and token embeddings,
with the source's `int8_mixed` language model and INT8 conditional decoder. Turbo `q4f16` selects
the four distinct Q4F16 graph exports. Static tokenizer/config artifacts and any ONNX external-data
companions are recorded with their own pinned URLs and integrity values.

The approved export sources are:

- `KitsuMate/chatterbox-nano-v2-onnx`
- `ResembleAI/chatterbox-turbo-ONNX`

The current exact resolved commits, requested revisions, artifact counts, and source provenance are
in `catalog/source.json`; every downloadable URL in `catalog/bundles.json` is pinned to the
corresponding 40-character revision. `catalog/seed.json` records stable intent only. Do not add
unverified metadata or hand-edit generated sizes, hashes, or URLs.

## Files

```text
catalog/seed.json
catalog/bundles.json
catalog/source.json
schemas/bundle-catalog.schema.json
scripts/check_catalog_integrity.py
```

`bundles.json` is generated from the seed by the Chatterbox catalog tooling in OnnxVoice. The
schema describes the normalized public contract. The integrity script checks supported IDs and
profiles, source/revision consistency, safe paths, positive sizes, lowercase SHA-256 values,
revision-pinned URLs, graph/external-data pairing, static artifact roles, and the data-only
repository boundary.

## Build and verify

From a checkout with the Chatterbox-enabled OnnxVoice source installed:

```bash
python scripts/check_catalog_integrity.py
onnxvoice catalog chatterbox verify \
  --catalog catalog/bundles.json \
  --source catalog/source.json

onnxvoice catalog chatterbox build \
  --seed catalog/seed.json \
  --output catalog/bundles.json \
  --source-output catalog/source.json
```

A deterministic rebuild at unchanged upstream revisions produces byte-identical JSON. The
scheduled refresh workflow builds twice, compares both catalog files, verifies the result, and
commits only generated catalog/provenance changes.

The CI workflows currently check out the OnnxVoice `main` source so this repository can be
validated before the first public OnnxVoice release containing Chatterbox support. Once that
release exists, pin the workflows to its released version as described by the implementation
plan; do not guess a version before publication.

## Licensing and attribution

Repository-authored metadata, scripts, schema, and documentation are licensed under Apache-2.0.
The model, tokenizer, configuration, and external-data artifacts remain subject to the terms and
attribution of their respective upstream snapshots. This repository does not relicense or
redistribute those files. See `NOTICE` and the exact source URLs for provenance.
