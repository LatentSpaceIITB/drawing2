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

## Phase 2 Direction

Move toward the realism of `data/PID2Graph OPEN100` by adding:
- richer equipment geometry
- larger symbol catalog
- more natural engineering composition
- variable page sizes and real-style title blocks
- less templated layout grammar
