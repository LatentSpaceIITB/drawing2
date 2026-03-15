# OPEN100 Structural Generation

This document describes the new large-scale generation path built on top of `data/PID2Graph OPEN100`.

## Why This Branch Exists

The original phase-1 pipeline targeted the templated synthetic dataset style in `data/Dataset PID`.

To avoid destabilizing that baseline, the OPEN100 methodology is developed on a separate git branch:
- `feature/open100-structural-synth`

The intent is to keep methodology experiments isolated so we can compare approaches later.

## Core Idea

- use `PID2Graph OPEN100` as the structural seed source
- keep the current synthetic sheet renderer first
- generate many variants by changing structure, density, motifs, and labels
- enforce balanced complexity and reject near-duplicates
- prioritize visually rich outputs, so the default production path is now dense-only

## Input Source

Seed directory:
- `data/PID2Graph OPEN100`

The OPEN100 seeds contain more realistic structural classes such as:
- `tank`
- `pump`
- `inlet/outlet`
- more natural trunk/branch arrangements

## What Changed In Code

### Parsing and metadata
- `src/pidgen/seed_parser.py`
- detects source dataset
- stores source page bounds and content bounds
- stores anchor sides for `inlet/outlet` nodes

### OPEN100-aware perturbation
- `src/pidgen/perturb.py`
- normalizes variable OPEN100 pages into the synthetic content frame
- preserves inlet/outlet border attachment semantics
- supports profile-driven mutation (`simple`, `medium`, `dense`)
- prunes leaf branches for simple outputs
- inserts inline nodes
- injects additional branch motifs
- retunes solid/non-solid edge ratios

### Motif enrichment
- `src/pidgen/motifs.py`
- adds small reusable branch patterns such as:
  - valve + instrumentation taps
  - pump chains
  - tank branches
  - terminal stubs

### Complexity control
- `src/pidgen/complexity.py`
- defines generation profiles
- computes complexity scores
- assigns complexity buckets
- includes two dense-oriented internal subprofiles:
  - `dense_standard`
  - `dense_heavy`

### Diversity control
- `src/pidgen/diversity.py`
- computes graph fingerprints using WL hashing
- computes image hashes using average hash and difference hash
- rejects near-duplicate outputs

### Rendering support for OPEN100 classes
- `src/pidgen/renderers.py`
- now renders:
  - `tank`
  - `pump`
  - `inlet/outlet`
- still uses the synthetic phase-1 sheet style

## Batch Generator

Main script:
- `scripts/generate_open100_batch.py`

Default config:
- `configs/open100_batch.json`

The batch generator:
- scans the OPEN100 seed folder
- derives complexity thresholds from the seed set
- chooses seed files and requested profiles
- generates candidate scenes
- checks actual complexity bucket
- rejects graph/image near-duplicates
- writes accepted outputs and manifests

## Complexity Balancing

The default production quota plan is now:
- `100%` dense

This matches the current objective: very rich, very diverse synthetic pages.

The generator still supports `simple` and `medium` internally, but the default OPEN100 batch config now focuses only on dense pages.

Within the dense bucket, the batch generator mixes two internal dense profiles:
- `dense_standard`
- `dense_heavy`

This keeps dense-only runs from collapsing into one narrow visual pattern.

Configured in:
- `configs/open100_batch.json`

The generator can accept a candidate into a different bucket than the requested one only if that bucket is enabled in the current config. In the dense-only default, this effectively means only dense candidates are accepted.

## Reproducibility

Large runs are controlled by:
- master seed
- input seed folder
- complexity quota plan
- target count
- max attempts

Each accepted sample records:
- requested profile
- accepted bucket
- source seed path
- image hashes
- SHA256 hashes
- scene statistics

## Commands

Generate one OPEN100-based sample:

```bash
python scripts/generate_phase1.py --seed-graphml "/home/pinak/MedClip/drawing2/data/PID2Graph OPEN100/0.graphml" --profile medium --stem open100_seed0_medium --output-dir outputs/open100_single
```

Generate the default dense-only large batch:

```bash
python scripts/generate_open100_batch.py --config configs/open100_batch.json
```

Generate a smaller pilot batch:

```bash
python scripts/generate_open100_batch.py --config configs/open100_batch.json --output-dir outputs/open100_pilot --target-count 24 --max-attempts 600
```

## Practical Note

The config targets `1000` dense samples by default, but in practice you should start with a pilot batch, inspect diversity, and then scale up.
