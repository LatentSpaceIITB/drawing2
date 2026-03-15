# Phase-1 Architecture

## Design Goal

Generate synthetic sheets that look like `data/Dataset PID` while keeping the pipeline deterministic and extensible.

## Pipeline Stages

1. Seed parsing
   - file: `src/pidgen/seed_parser.py`
   - reads GraphML nodes and edges into a canonical scene model

2. Scene model
   - file: `src/pidgen/scene.py`
   - defines `BoundingBox`, `Node`, `Edge`, and `Scene`

3. Layout perturbation
   - file: `src/pidgen/perturb.py`
   - remaps grid coordinates
   - rescales node boxes by class
   - inserts additional inline nodes on selected long edges
   - retunes some edges to `non-solid`

4. Synthetic annotation
   - file: `src/pidgen/annotate.py`
   - generates notes, project metadata, instrument labels, valve tags, and pipe specs
   - assigns visual variants to symbols

5. Rendering
   - file: `src/pidgen/renderers.py`
   - renders a PNG sheet using Pillow
   - emits SVG from the same geometry

6. Annotation export
   - file: `src/pidgen/export_graphml.py`
   - exports aligned GraphML for the generated scene

7. Pipeline orchestration
   - file: `src/pidgen/pipeline.py`
   - drives generation and writes a manifest JSON

## Output Alignment Principle

The same scene graph is the source of truth for:
- node boxes
- edge connectivity
- raster rendering
- vector rendering
- manifest statistics

## Symbol Coverage In Phase 1

Node classes currently supported:
- `background`
- `connector`
- `crossing`
- `valve`
- `instrumentation`
- `arrow`
- `general`

Current `general` variants include:
- vessel
- unit box
- pump box
- double box
- stacked box
- heater
- reducer
- orifice
- terminal
- tag box
- station box

Current `instrumentation` variants include:
- round
- panel
- inline
- dashed

Current `valve` variants include:
- gate
- control
- check
- relief

## Reproducibility Mechanisms

- explicit seed passed through perturbation and annotation
- batch config with fixed job list
- per-sample manifest JSON
- file hashes in manifest
- pinned dependencies
