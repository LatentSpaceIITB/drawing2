# High-Level Design

This document explains the phase-1 generator as an input-to-output pipeline.

The most important design choice is this:

- we do not generate `PNG`, `SVG`, and `GraphML` independently
- we build one in-memory `Scene`
- every output is produced from that same `Scene`

That is why the outputs stay aligned.

## Inputs And Outputs

### Inputs

1. Seed structure
   - example: `data/Dataset PID/0.graphml`
   - contains node classes, boxes, and graph connectivity

2. Integer seed
   - example: `1401`
   - controls perturbation, symbol choices, and generated text

3. Run parameters
   - output directory
   - file stem

### Outputs

1. Raster sheet
   - `*.png`

2. Vector sheet
   - `*.svg`

3. Annotation graph
   - `*.graphml`

4. Reproducibility manifest
   - `*.manifest.json`
   - contains seed, output paths, hashes, runtime info, and scene stats

## Core Data Object

The central object is `Scene` from `src/pidgen/scene.py`.

`Scene` contains:
- canvas size
- `nodes`
- `edges`
- metadata

Each `Node` has:
- `id`
- `label`
- `bbox`
- metadata such as text and visual variant

Each `Edge` has:
- `source`
- `target`
- style such as `solid` or `non-solid`
- optional metadata such as pipe label text

## Data Flow

```mermaid
flowchart LR
    A[Seed GraphML\nexample: data/Dataset PID/0.graphml] --> B[parse_graphml\nsrc/pidgen/seed_parser.py]
    B --> C[Scene\nseed geometry + connectivity]
    D[Integer Seed\nexample: 1401] --> E[perturb_scene\nsrc/pidgen/perturb.py]
    C --> E
    E --> F[Scene\nperturbed geometry + some topology edits]
    D --> G[annotate_scene\nsrc/pidgen/annotate.py]
    F --> G
    G --> H[Scene\nvisual variants + synthetic text + title block metadata]
    H --> I[render_png\nsrc/pidgen/renderers.py]
    H --> J[render_svg\nsrc/pidgen/renderers.py]
    H --> K[export_graphml\nsrc/pidgen/export_graphml.py]
    H --> L[scene_summary\nsrc/pidgen/stats.py]
    I --> M[PNG]
    J --> N[SVG]
    K --> O[GraphML]
    L --> P[stats]
    M --> Q[manifest.json]
    N --> Q
    O --> Q
    P --> Q
```

## Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI as scripts/generate_phase1.py
    participant Pipe as generate_outputs()
    participant Parser as parse_graphml()
    participant Perturb as perturb_scene()
    participant Annotate as annotate_scene()
    participant Render as render_png()/render_svg()
    participant Export as export_graphml()
    participant Stats as scene_summary()

    User->>CLI: run with seed_graphml + seed + stem
    CLI->>Pipe: generate_outputs(seed_graphml, output_dir, stem, seed)
    Pipe->>Parser: parse_graphml(seed_graphml)
    Parser-->>Pipe: Scene(seed nodes + edges + boxes)

    Pipe->>Perturb: perturb_scene(scene, seed)
    Note over Perturb: remap grid\nrescale boxes\ninsert inline nodes\nretune some edge styles
    Perturb-->>Pipe: Scene(perturbed)

    Pipe->>Annotate: annotate_scene(scene, seed)
    Note over Annotate: add title block values\nadd notes\nadd instrument labels\nadd valve tags\nassign symbol variants
    Annotate-->>Pipe: Scene(annotated)

    par Output Rendering
        Pipe->>Render: render_png(scene, png_path)
        Pipe->>Render: render_svg(scene, svg_path)
        Pipe->>Export: export_graphml(scene, graphml_path)
        Pipe->>Stats: scene_summary(scene)
    end

    Note over Pipe: hash files\ncollect runtime info\nwrite manifest
    Pipe-->>CLI: manifest dictionary
    CLI-->>User: paths to PNG, SVG, GraphML, manifest
```

## What Each Stage Actually Changes

### 1. `parse_graphml()`

File: `src/pidgen/seed_parser.py`

What goes in:
- raw GraphML XML

What comes out:
- a `Scene` with seed node boxes and edges

What it does not do:
- no text generation
- no rendering
- no randomness

### 2. `perturb_scene()`

File: `src/pidgen/perturb.py`

What goes in:
- seed `Scene`
- integer seed

What it changes:
- remaps x/y grid positions
- rescales node bounding boxes by class
- inserts a few new inline symbols on long edges
- changes some solid edges to non-solid
- resets the fixed background panel boxes

What comes out:
- a new `Scene` that still looks like the same family, but not like an exact copy

### 3. `annotate_scene()`

File: `src/pidgen/annotate.py`

What goes in:
- perturbed `Scene`
- integer seed

What it adds:
- title block metadata
- notes panel text
- project/unit/drawing identifiers
- node variants
- instrument text
- valve tags
- edge labels such as line specs

Important point:
- this stage mostly adds semantic and visual metadata
- it does not create final image files

### 4. `render_png()` and `render_svg()`

File: `src/pidgen/renderers.py`

What goes in:
- annotated `Scene`

What it does:
- draws the synthetic sheet template
- draws pipes from scene connectivity
- draws symbols from node classes and variants
- draws notes, title block, and labels

Why two renderers exist:
- `PNG` is easy to inspect and use as raster output
- `SVG` preserves vector structure for downstream editing or inspection

### 5. `export_graphml()`

File: `src/pidgen/export_graphml.py`

What goes in:
- the same annotated `Scene`

What it does:
- writes node boxes and edges back to GraphML

Why this matters:
- the annotation graph is synchronized with the rendered page because both come from the same geometry

### 6. `generate_outputs()`

File: `src/pidgen/pipeline.py`

This is the orchestration layer.

It:
- calls parse
- calls perturb
- calls annotate
- calls both renderers
- calls GraphML export
- computes scene stats
- computes SHA256 hashes
- writes manifest JSON

## Artifact Lineage

```mermaid
flowchart TD
    A[seed GraphML] --> B[Scene after parse]
    B --> C[Scene after perturb]
    C --> D[Scene after annotate]
    D --> E[PNG]
    D --> F[SVG]
    D --> G[GraphML]
    D --> H[Stats]
    E --> I[Manifest]
    F --> I
    G --> I
    H --> I
```

## Why This Design Is Good For Synthetic Data

Because all outputs come from one scene graph, we get:
- box/image consistency
- deterministic generation with a fixed seed
- easy extension with new symbol classes
- a clean place to add realism later

## Mental Model

If you want a simple mental model, think of the pipeline like this:

1. read one synthetic graph template
2. reshape it a bit
3. decorate it with new engineering-looking metadata
4. draw it twice (`PNG`, `SVG`)
5. export the same structure as `GraphML`
6. record exactly what was generated in a manifest

That is the full input-to-output story.
