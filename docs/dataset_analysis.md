# Dataset Analysis

## Target Datasets

### `data/PID2Graph OPEN100`
- sample count: 12 pairs
- visually closer to real engineering drawings
- variable page sizes
- more expressive equipment drawings
- more natural composition and spacing
- better long-term realism target

### `data/Dataset PID`
- sample count: 500 pairs
- highly templated synthetic layout
- fixed canvas around `7168 x 4561/4562`
- gray background sheet
- dashed outer frame
- fixed right-side notes and title block
- orthogonal pipes and repetitive symbol vocabulary

## Why `Dataset PID` Was Chosen For Phase 1

It has a stable, learnable rendering grammar:
- fixed page template
- limited symbol set
- consistent margins
- repeated title-block structure
- simple black-on-gray line art
- many examples for future statistical tuning

## Class Distribution Observed

The synthetic dataset is dominated by:
- `connector`
- `crossing`
- `valve`
- `general`
- `instrumentation`
- `arrow`
- `background`

This makes it practical to focus phase 1 on a compact symbol library rather than a full engineering standards catalog.

## Typical Structural Characteristics

- mostly orthogonal routing
- mostly solid edges, some non-solid signal/instrument lines
- many small inline symbols on long horizontal and vertical trunks
- frequent label text on longer line segments
- title-block text is decorative and style-critical, but not the main annotation target

## Implication For Generation

A vector-first generator can be built around:
- scene graph
- symbol primitives
- route perturbation
- text synthesis
- synchronized export to image and annotation graph
