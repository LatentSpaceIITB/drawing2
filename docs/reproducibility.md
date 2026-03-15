# Reproducibility Guide

## Runtime Baseline

This pipeline was built and verified with:
- Python `3.13.11`
- Pillow `12.1.0`
- networkx `3.6.1`

Pinned dependency files:
- `requirements.txt`
- `pyproject.toml`

## Determinism Rules

Reproducibility depends on two inputs:
- the seed GraphML file
- the integer generation seed

Given the same:
- source GraphML
- code revision
- Python version
- dependencies
- generation seed

the output should be reproducible.

## One-Sample Reproduction

```bash
python scripts/generate_phase1.py --seed-graphml "data/Dataset PID/0.graphml" --seed 1401 --stem phase1_sample_seed1401
```

Artifacts written:
- `outputs/phase1/phase1_sample_seed1401.png`
- `outputs/phase1/phase1_sample_seed1401.svg`
- `outputs/phase1/phase1_sample_seed1401.graphml`
- `outputs/phase1/phase1_sample_seed1401.manifest.json`

## Batch Reproduction

```bash
python scripts/generate_phase1_batch.py --config configs/phase1_batch.json
```

This creates:
- multiple sample triplets
- one manifest per sample
- `outputs/phase1_batch/batch_manifest.json`

## Verification

```bash
make verify
```

The verification step checks:
- Python syntax compilation
- one end-to-end generation run
- file existence
- GraphML node and edge counts against manifest statistics

## Manifest Contents

Each per-sample manifest records:
- seed GraphML path
- generation seed
- output file paths
- SHA256 hashes
- runtime information
- scene summary statistics
