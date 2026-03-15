# Project Journal

## Initial Problem

We wanted to evaluate whether P&ID diagrams could be generated programmatically, ideally as vector-first outputs, and whether a first synthetic result could be made to look like the examples in `data/Dataset PID`.

## Dataset Review

Two datasets were reviewed:

1. `data/PID2Graph OPEN100`
   - real-world
   - fewer samples
   - more natural layout variation
   - richer equipment drawings
   - white engineering sheet style

2. `data/Dataset PID`
   - synthetic
   - many samples
   - fixed canvas and repeated title-block layout
   - orthogonal routing with repeated visual grammar
   - gray paper background and dashed border

## Key Decision

We chose a two-phase plan:

- Phase 1: reproduce the synthetic visual language of `data/Dataset PID`
- Phase 2: move toward the richer and less templated realism of `data/PID2Graph OPEN100`

## Strategy Chosen

For phase 1 we selected a hybrid-bootstrap method instead of a fully procedural grammar from scratch.

Why:
- it is the fastest route to a believable first result,
- the seed GraphML already contains useful structural information,
- the pipeline can still be built in a clean, vector-first way.

## Output Contract

The first milestone was fixed as:
- `PNG`
- `SVG`
- `GraphML`

All three outputs come from the same internal scene graph.

## Phase-1 Implementation Summary

The implemented pipeline now does the following:
- parse a seed GraphML from `data/Dataset PID`
- perturb its grid, spacing, node sizes, and some topology
- insert additional inline nodes on selected long edges
- mutate a subset of line styles to dashed/non-solid
- synthesize engineering-like notes, tags, pipe labels, and title-block values
- render a synthetic sheet in both raster and vector form
- export GraphML aligned with the perturbed scene
- write a manifest JSON for each generated sample

## Improvements Added After First Prototype

The first prototype was extended with:
- more `general` symbol variants
- more `instrumentation` and `valve` variants
- stronger perturbation logic
- reproducible batch generation
- verification tooling
- pinned dependency files and build commands
- project documentation
