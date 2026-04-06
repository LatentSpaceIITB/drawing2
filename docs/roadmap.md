# Roadmap

## Completed In Phase 1

- hybrid-bootstrap scene generation from synthetic GraphML
- synchronized `PNG`, `SVG`, `GraphML` outputs
- title block and notes panel rendering
- symbol variation for core synthetic classes
- batch generation
- manifests and verification
- reproducibility files and docs

## Next Phase-1 Improvements

1. Better route perturbation
   - branch insertion beyond inline nodes
   - more distinct spatial layouts from the same seed

2. Better visual fidelity
   - denser title-block microtext
   - more nuanced line-end symbols
   - more realistic instrument attachment patterns

3. Better dataset-scale generation
   - sampling across many seed graphml files
   - count controls by class distribution targets
   - optional output manifests in CSV/JSONL

## Phase 2: ML Experiment (Synthetic-to-Real Transfer)

Full plan is in `docs/research_plan.md`.

Summary:
- **Core claim**: a model trained only on synthetic P&IDs (generated from 12 real
  seeds) can generalise to unseen real-world P&ID graph extraction
- **Model**: Relationformer (ECCV 2022) — same model used by Stürmer et al. on
  OPEN-100, giving a direct comparison anchor
- **Real data**: 12 OPEN-100 images used for evaluation only (5-fold CV, 3 seeds)
- **Synthetic data**: 165 samples from `outputs/open100_dense_1000_v2/`
- **Experiments**: oracle / synth-only / mixed / scale-ablation / baseline-replication
- **Target output**: research paper on synthetic-to-real transfer for P&ID digitization

### Phase 2 Implementation Steps

1. Data preprocessing (`scripts/preprocess/`)
   - connector-collapse GraphML files (remove routing waypoints, keep physical components)
   - convert collapsed GraphML to COCO-style JSON for Relationformer
   - generate 5-fold CV split indices

2. Model setup (`third_party/relationformer/` + `scripts/train/`)
   - clone Relationformer, write P&ID data loader
   - configure for 4× A6000 multi-GPU training

3. Run experiments E1–E5 (see research_plan.md Section 6)

4. Analysis and paper writing

## Phase 2 Generator Improvements (Visual Fidelity)

These are lower priority than the ML experiment but improve synthetic data quality
for future iterations:
- richer equipment geometry and larger symbol catalog
- more natural engineering composition
- variable page sizes and real-style title blocks
- less templated layout grammar
