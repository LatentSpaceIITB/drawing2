# Phase-1 Synthetic P&ID Generator

This directory contains a reproducible phase-1 pipeline for generating synthetic P&ID-like sheets in the visual style of `data/Dataset PID`.

The generator is hybrid-bootstrap based:
- it parses an existing synthetic `GraphML` seed,
- perturbs the layout and topology,
- annotates the scene with fresh synthetic engineering text,
- renders aligned `PNG`, `SVG`, and `GraphML` outputs.

## Why This Exists

The original objective was to explore whether P&ID diagrams could be generated programmatically, with a first milestone of creating synthetic pages that look like the examples in `data/Dataset PID`.

After comparing:
- `data/PID2Graph OPEN100` (real-world, more organic, more diverse)
- `data/Dataset PID` (synthetic, strongly templated, easier phase-1 target)

we chose to start with the synthetic style because it is visually regular and easier to reproduce deterministically.

## Current Outputs

Single-sample output:
- `outputs/phase1/`

Batch output:
- `outputs/phase1_batch/`

Verification output:
- `outputs/verify/`

## Quick Start

Install pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate one sample:

```bash
python scripts/generate_phase1.py --stem phase1_sample_seed1401 --seed 1401
```

Generate the default reproducible batch:

```bash
python scripts/generate_phase1_batch.py --config configs/phase1_batch.json
```

Generate the default dense-only OPEN100-structural batch:

```bash
python scripts/generate_open100_batch.py --config configs/open100_batch.json
```

Run syntax + pipeline verification:

```bash
make verify
```

## Files

Core package:
- `src/pidgen/scene.py`
- `src/pidgen/seed_parser.py`
- `src/pidgen/perturb.py`
- `src/pidgen/annotate.py`
- `src/pidgen/renderers.py`
- `src/pidgen/export_graphml.py`
- `src/pidgen/pipeline.py`
- `src/pidgen/stats.py`

Scripts:
- `scripts/generate_phase1.py`
- `scripts/generate_phase1_batch.py`
- `scripts/verify_phase1.py`

Reproducibility:
- `requirements.txt`
- `pyproject.toml`
- `Makefile`
- `configs/phase1_batch.json`
- `configs/open100_batch.json`

Documentation:
- `docs/project_journal.md`
- `docs/dataset_analysis.md`
- `docs/high_level_design.md`
- `docs/open100_structural_generation.md`
- `docs/phase1_architecture.md`
- `docs/reproducibility.md`
- `docs/roadmap.md`
