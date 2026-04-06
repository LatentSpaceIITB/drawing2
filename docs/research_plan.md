# Research Plan: Synthetic-to-Real P&ID Graph Extraction

## Status

Active — last updated 2026-03-15.

This document records all decisions, rationale, and the full implementation plan
for the ML research direction of this project. It is the authoritative reference
for anyone starting work on Phase 2 (the ML experiment).

---

## 1. Motivation and Context

### The industrial problem

Piping and Instrumentation Diagrams (P&IDs) are the primary engineering document
for process plants (oil and gas, chemical, power generation). They encode every
instrument, valve, pump, and pipe connection in a facility. Maintaining a digital,
queryable graph of a plant's P&IDs is critical for safety audits, maintenance
scheduling, and digital-twin workflows.

Today, most P&IDs exist only as scanned PDFs or raster images. Manual digitization
is slow and expensive. Automated digitization — detecting components and their
connections — is a long-standing open problem.

### The data scarcity problem

The core obstacle is that **real-world P&ID data is extremely scarce and proprietary**.
No operating plant will share their engineering drawings externally; they contain
confidential process information. The only publicly available annotated dataset with
graph-level ground truth is the OPEN-100 benchmark (12 images), released by
Stürmer et al. (2025, IEEE DSAA).

### What we have

- **12 real-world P&ID images** with GraphML graph annotations (`data/PID2Graph OPEN100`)
  — this is the same data as the Stürmer et al. public benchmark.
- **165 synthetic P&ID images** with GraphML annotations generated from those 12
  seeds (`outputs/open100_dense_1000_v2`), using our Phase-1 pipeline with WL-hash
  and perceptual-hash deduplication (165 accepted from ~12,000 attempts).

### The research contribution

> We demonstrate that a model trained exclusively on synthetic P&IDs generated from
> a small set of real-world seeds achieves competitive performance on unseen real-world
> P&ID graph extraction — establishing a practical pathway for P&ID digitization in
> data-scarce industrial settings.

This is a **synthetic-to-real transfer** paper. The selling point is that industries
do not need to share proprietary plant data: a small seed set of reference P&IDs is
enough to generate a training corpus.

### Prior work and our baseline

The closest prior work is:

> Stürmer, Graumann, Koch. "From Engineering Diagrams to Graphs: Digitizing P&IDs
> with Transformers." IEEE DSAA 2025. arXiv:2411.13929.

They apply Relationformer to the OPEN-100 dataset and achieve >25% improvement in
edge detection over a modular baseline. They do not address data scarcity or
synthetic augmentation. **Their results on OPEN-100 are our primary comparison
anchor.**

---

## 2. Problem Statement

### Formal definition

Given a P&ID raster image `I`, extract the process graph `G = (V, E)` where:

- `V` = set of detected physical components, each with:
  - a bounding box `(xmin, ymin, xmax, ymax)` in image coordinates
  - a class label from {`valve`, `pump`, `instrumentation`, `general`, `tank`,
    `arrow`, `inlet/outlet`}
- `E` = set of pipe/signal connections between components, each with:
  - a pair of endpoint node IDs `(u, v)`
  - an edge label from {`solid`, `non-solid`}

**Connector and crossing nodes are removed during preprocessing** (see Section 3).
They are routing waypoints, not physical components. After collapsing, the graph
contains only the 7 physically meaningful node classes listed above.

### What this is NOT

- Not OCR / text extraction (instrument tag recognition is out of scope for now).
- Not line tracing (we predict edges as direct component-to-component connections,
  not as pixel-level paths).
- Not a multi-page problem (each image is treated independently).

---

## 3. The Connector-Collapsing Decision

### What connector and crossing nodes are

In the raw GraphML annotations, `connector` nodes (~8×8 px) are routing waypoints
placed at every pipe bend, branch point, and junction. `crossing` nodes (~8×8 px)
mark places where two pipes cross without connecting.

A pipe path between a valve and a pump is represented as:

```
valve — conn — conn — conn — pump
```

rather than the physically meaningful:

```
valve —(solid)— pump
```

Connector and crossing nodes together account for **50–65% of all nodes** in the
dataset but carry no physical meaning beyond pipe routing geometry.

### Decision: collapse connectors

We collapse connector and crossing nodes before training and evaluation.

**Arguments for collapsing:**

1. The industrially meaningful question is "what connects to what?" — not "how many
   waypoints are on the pipe?".
2. Evaluation becomes interpretable: matching a predicted edge (valve→pump) against
   ground truth is clean. Matching predicted waypoint positions is fragile and
   penalizes topologically correct but geometrically shifted predictions.
3. Removes severe class imbalance: without collapsing, ~50–65% of the training
   signal is spent on 8×8 px artifacts instead of meaningful symbols.
4. Makes the problem tractable for Relationformer, which was designed for graphs
   with ~10–100 meaningful nodes, not 500+ waypoints.
5. Stürmer et al. almost certainly evaluate at the symbol-level graph (not waypoint
   graph) — this keeps our results directly comparable.

**Trade-offs acknowledged:**

1. Pipe routing geometry (the spatial path of a pipe) is lost. For applications
   that need isometric layout or pipe-length calculation, the waypoints carry
   value. This is treated as a separate downstream problem (line tracing), which
   classical CV handles well on clean P&IDs.
2. T-junctions (degree-3 connectors) need careful handling — see algorithm below.
3. `crossing` nodes require special handling: they mark non-connecting overlaps and
   must NOT be collapsed as if they were junctions.

**Decision on crossing nodes:** Remove crossing nodes entirely. Do not create edges
through them. The two pipes that cross are independent paths; they should already
be connected via other routes in the graph.

### Collapse algorithm

```
For each GraphML graph:
  1. Identify all connector nodes (label == "connector")
     and crossing nodes (label == "crossing").
  2. Remove crossing nodes and all their incident edges entirely.
  3. For each connected component in the subgraph induced by connector nodes:
     a. Find all physical-component nodes (non-connector, non-crossing, non-background)
        that are adjacent to any node in this connector component.
     b. Collect all edges in this connector component's path (their labels).
     c. Compute the majority edge label (solid vs non-solid).
     d. For every pair of physical-component neighbors of this connector component,
        add a direct edge with the majority label.
  4. Remove all connector nodes and their original edges.
  5. Remove all background nodes (page border artifacts).
  6. Deduplicate edges: if two physical components are connected by multiple
     collapsed paths, keep one edge (majority label across all paths).
```

**Edge label inheritance rule:** majority vote along the connector chain. If all
edges in the chain are `solid`, the collapsed edge is `solid`. If mixed, the
majority wins. Ties default to `solid` (more common class).

**T-junction handling:** A degree-3 connector is a branch point. The collapse
algorithm handles this naturally: the connector component connected to 3 physical
components produces 3 pairwise edges (a triangle in the collapsed graph). This
correctly represents a branching pipe.

---

## 4. Dataset Inventory

### Real-world data (evaluation set)

| Property | Value |
|---|---|
| Location | `data/PID2Graph OPEN100/` |
| Samples | 12 (paired .png + .graphml, named 0–11) |
| Source | OPEN-100 benchmark, Stürmer et al. (2025) |
| Canvas | ~2288 × 1528 px (variable) |
| Node coordinate type | float doubles in d1–d4 |
| Typical node count | ~100–250 per image |
| Edge labels | `solid` / `non-solid` |
| Use in experiments | Evaluation only (5-fold CV) |

### Synthetic data (training set)

| Property | Value |
|---|---|
| Location | `outputs/open100_dense_1000_v2/` |
| Samples | 165 (quad: .png + .graphml + .svg + .manifest.json) |
| Generation | Phase-1 pipeline, seeded from the 12 real GraphML files |
| Canvas | 7168 × 4562 px (fixed) |
| Node coordinate type | long integers in d1–d4 |
| Typical node count | ~380–730 per image |
| Edge labels | `solid` / `non-solid` |
| Deduplication | WL-hash (graph) + perceptual hash (image) — 165 accepted from ~12,000 |
| Use in experiments | Training only |

### Important coordinate system difference

Real-world GraphML stores primary node bounding boxes as **float doubles** in keys
`d1–d4` (xmin, xmax, ymin, ymax order) while connector/crossing nodes use integer
keys `d5–d8`.

Synthetic GraphML stores primary node bounding boxes as **long integers** in keys
`d1–d4` (xmin, ymin, xmax, ymax order) while other nodes use float doubles in
`d5–d8`.

The preprocessing pipeline normalises both to a canonical format:
`(xmin, ymin, xmax, ymax)` as floats, before any downstream use.

### Evaluation strategy

- All 12 real images are used for evaluation via **5-fold cross-validation**.
  Each fold holds out ~2–3 real images.
- All 165 synthetic images are used for training in the synthetic-only experiments.
- No synthetic images appear in any evaluation fold.
- Each experiment is run with **3 random seeds**; results reported as mean ± std.

---

## 5. Model Choice

### Primary model: Relationformer

> Shit et al. "Relationformer: A Unified Framework for Image-to-Graph Generation."
> ECCV 2022. arXiv:2203.10202. GitHub: https://github.com/suprosanna/relationformer

**Why Relationformer:**

1. Stürmer et al. directly applied it to OPEN-100 and published results — we get a
   free comparison anchor on our exact evaluation set.
2. The task formulation matches perfectly: image → nodes (bbox + class) + adjacency
   matrix with edge labels. No adaptation of output format needed.
3. Built on DETR/Deformable-DETR, so transfer from ImageNet-pretrained backbone is
   straightforward.
4. The rln-token mechanism (one relation token combined pairwise with object tokens)
   naturally produces an undirected edge prediction, which matches our undirected
   P&ID graph.

**Known limitations:**

- Small codebase (~4 commits, last updated 2022). May need minor maintenance.
- No pretrained weights on diagram data — must train from scratch or from COCO
  pretrained DETR.
- Designed for small-to-medium node counts; may struggle with very dense
  synthetic images (730 nodes). Connector collapsing mitigates this.

### Comparison baselines

| Model | Role | Notes |
|---|---|---|
| Modular pipeline (symbol detect + line trace) | Baseline | Stürmer et al.'s modular baseline; classical approach |
| Relationformer (Stürmer et al. setup) | Prior-work anchor | Their reported numbers on OPEN-100 |
| EGTR (Im et al., CVPR 2024) | Secondary ML comparison | Better architecture but no P&ID prior work; optional |

### Why not EGTR as primary

EGTR (arXiv:2404.02072, CVPR 2024 best paper candidate) is architecturally superior
— it extracts relation graphs directly from DETR decoder self-attention matrices
without bespoke relation tokens, resulting in a lighter and better-performing model
for scene graph generation.

However, no prior P&ID work uses EGTR. Using Relationformer as primary gives us
direct numerical comparability to Stürmer et al. EGTR can be added as a secondary
experiment to show architectural sensitivity.

### Why not RelTR

RelTR (arXiv:2201.11460) uses coupled subject-object queries that predict sparse
triplets directly. For P&IDs, we want a complete node inventory first (including
nodes with no visible pipe connection in a sub-crop), then edges. RelTR's
triplet-centric decoder can miss such isolated nodes.

---

## 6. Experiment Table

| ID | Name | Train set | Eval set | Purpose |
|---|---|---|---|---|
| E1 | Oracle | 12 real (5-fold CV) | 12 real (held-out fold) | Upper bound; best achievable with real data only |
| E2 | Synth-only | 165 synthetic | 12 real (all, 5-fold) | **Core claim**: does synthetic training transfer? |
| E3 | Mixed | 165 synth + partial real | 12 real (held-out fold) | Practical best-practice setting |
| E4 | Scale ablation | 40 / 80 / 120 / 165 synth | 12 real (all, 5-fold) | Does more synthetic data monotonically help? |
| E5 | Baseline replication | Stürmer et al. setup | OPEN-100 | Reproducibility anchor; validates our eval framework |

**E2 is the central experiment.** The paper's contribution stands or falls on
whether synthetic training (E2) closes the gap toward oracle (E1).

**E4 provides the scaling story**: if accuracy improves monotonically with synthetic
volume, we have motivation to generate more data. This result informs whether to
scale the synthetic corpus beyond 165 samples.

---

## 7. Metrics

Following Stürmer et al. and standard graph extraction evaluation:

### Node detection

- **mAP@0.5 IoU** per class + mean across all 7 physical classes
- A predicted node is correct if IoU with a ground-truth box ≥ 0.5 and the class
  label matches.

### Edge detection

- **Precision / Recall / F1** on (node_i, edge_label, node_j) triplets.
- A predicted edge is correct only if **both** endpoint nodes are correctly detected
  (matched to ground-truth nodes with IoU ≥ 0.5 and correct class) **and** the
  edge label matches.
- This is the standard "relation detection" metric from scene graph generation
  literature (used by both EGTR and Relationformer papers).

### Graph-level (supplementary)

- **Graph Edit Distance (GED)** between predicted and ground-truth collapsed graph.
- Reported as supplementary; not the primary ranking metric due to NP-hardness for
  large graphs (approximated).

### Reporting convention

- All metrics reported as **mean ± std** across:
  - 5 cross-validation folds (for experiments using real eval data)
  - 3 random training seeds
- Total runs per experiment: 5 folds × 3 seeds = 15 training runs.

---

## 8. Implementation Phases

### Phase 2a: Data Preprocessing

**Goal:** Produce clean, model-ready datasets from raw GraphML files.

Steps:
1. **Connector collapse script** (`scripts/preprocess/collapse_connectors.py`)
   - Input: a directory of GraphML files
   - Output: collapsed GraphML files (connector/crossing/background nodes removed,
     physical-component graph only)
   - Handles coordinate normalisation (real vs synthetic format difference)
   - Handles T-junctions and crossing nodes correctly

2. **GraphML → COCO-style JSON converter** (`scripts/preprocess/graphml_to_coco.py`)
   - Input: directory of collapsed GraphML files + matching PNG files
   - Output: COCO-format JSON with:
     - `images`: id, file_name, height, width
     - `annotations`: id, image_id, category_id, bbox [x,y,w,h], area
     - `relations`: id, image_id, subject_id, object_id, predicate_id
     - `categories`: the 7 physical node classes
     - `predicates`: solid (0), non-solid (1)
   - This is the format expected by Relationformer's data loader (with minor
     adaptation for the relations field).

3. **Dataset split generator** (`scripts/preprocess/split_dataset.py`)
   - Generates 5-fold CV split indices for the 12 real images
   - Saves as JSON for reproducibility
   - Ensures each fold has at least 2 images

### Phase 2b: Model Setup

**Goal:** Get Relationformer running on P&ID data.

Steps:
1. Clone Relationformer into `third_party/relationformer/`
2. Write a P&ID-specific data loader that reads our COCO-style JSON
3. Configure the node class vocabulary (7 classes) and edge vocabulary (2 classes)
4. Set up multi-GPU training config for 4× A6000 (using PyTorch DDP)
5. Write training and evaluation launch scripts in `scripts/train/` and
   `scripts/eval/`

### Phase 2c: Experiments

Run E1 → E5 in order. E5 (baseline replication) should be run first to validate
the eval framework before investing in the full experiment suite.

### Phase 2d: Analysis and Paper

1. Qualitative visualisation: overlay predicted graph on real P&ID images
2. Per-class error analysis: which symbol classes are hardest to detect?
3. Edge error analysis: which connection types are most often missed or spurious?
4. Failure case catalogue: images where the model fails and why
5. Draft paper sections in order: Experiments → Method → Introduction → Related Work

---

## 9. Infrastructure

| Resource | Spec |
|---|---|
| GPUs | 4× NVIDIA A6000 (48 GB each = 192 GB total VRAM) |
| Training framework | PyTorch + DDP (torch.distributed) |
| Python version | 3.13 (pinned in .python-version) |
| Package management | uv / pyproject.toml |
| Experiment tracking | TBD — MLflow or Weights & Biases |

---

## 10. Decisions Log

This section records every significant decision made, why it was made, and whether
it is still open or resolved.

| # | Decision | Status | Rationale |
|---|---|---|---|
| D1 | Collapse connector and crossing nodes before training | **Resolved** | Industrial meaning, evaluation cleanliness, class balance — see Section 3 |
| D2 | Edge label inheritance: majority vote along chain | **Resolved** | Majority is unambiguous, ties default to `solid` |
| D3 | Crossing nodes: remove entirely, do not create edges | **Resolved** | Crossings are non-connecting overlaps; bridging them would create false edges |
| D4 | Primary model: Relationformer | **Resolved** | Direct comparability to Stürmer et al. on OPEN-100 |
| D5 | Evaluation: 5-fold CV on 12 real images, 3 seeds | **Resolved** | Standard practice for small-dataset papers; variance must be reported |
| D6 | Synthetic dataset: use open100_dense_1000_v2 (165 samples) | **Resolved** | Most recent, highest-quality run with WL-hash + perceptual-hash deduplication |
| D7 | EGTR as secondary comparison | **Resolved** | Add if time permits; not required for core contribution |
| D8 | Scale synthetic corpus beyond 165 before experiments | **Resolved: No** | Start experiments with 165; E4 ablation will tell us if scaling is worth it |
| D9 | OCR / text feature integration | **Open** | Instrument labels in images could help; out of scope for initial paper |
| D10 | Coordinate normalisation strategy (real vs synthetic GraphML) | **Open** | Implement in preprocessing; validate visually before training |

---

## 11. Open Questions

1. **Coordinate scale mismatch**: Real images are ~2288×1528 px; synthetic are
   7168×4562 px. Relationformer likely resizes inputs. Confirm that bounding box
   coordinates are normalised consistently after resizing.

2. **Node count after collapsing**: Synthetic images have ~380–730 total nodes.
   After collapsing, the physical-component count will be much smaller (~80–200
   estimated). Confirm this is within Relationformer's query budget.

3. **Relationformer query count**: The default DETR query count is 100–300.
   If post-collapse graphs have >300 physical components per image, the query
   count needs to be increased.

4. **Background nodes**: Real GraphML has `background` nodes (large rectangles
   representing the page border/margins). These should be removed along with
   connectors in preprocessing. Confirm this.

5. **Arrow nodes**: The `arrow` class represents flow direction indicators.
   These are physically meaningful but spatially ambiguous (tiny, ~10×10 px,
   often near the pipe rather than on a component). Decide whether to include
   them in the node vocabulary or remove them.

6. **Evaluation metric alignment with Stürmer et al.**: Their paper does not
   describe their metric implementation in full detail. Reproduce their modular
   baseline results first (E5) to confirm our metric code is aligned before
   comparing against their Relationformer numbers.

---

## 12. Implementation Bug Log

This section records bugs found during implementation that would have silently
invalidated all evaluation results. Captured here as a record and for any future
reader reproducing the work.

| # | File | Bug | Impact | Fix | Discovered |
|---|---|---|---|---|---|
| B1 | `dataset_pid.py` | `RandomSizeCrop(384,600)` in training augmentation | Predicted boxes 9.5× too large relative to GT → IoU always 0 → R@K=0 for 50 epochs | Removed crop; use `RandomResize` + `RandomHorizontalFlip` only | During first training run (epoch 25 analysis) |
| B2 | `train_pid.py` `evaluate()` | GT boxes manually converted xyxy→cxcywh before passing to evaluator, but they are already cxcywh after `Normalize` transform | Evaluator received scrambled coordinates | Pass `target["boxes"]` directly — no conversion needed | During first training run debugging |
| B3 | `train_pid.py` `evaluate()` | `pred_class + 1` made pred labels 1-indexed while GT labels are 0-indexed | Class match always failed → no triplet ever counted as correct → R@K=0 | Remove `+1`; pass 0-indexed labels | Found via perfect-prediction sanity check |
| B4 | `losses.py` | `F.cross_entropy` inside `if len(freq_dist) > 0` block | Edge loss always `None` when `FREQ_BIAS=False` → training crashed | Move cross_entropy outside the condition | Found during first DDP launch |
| B5 | `util/misc.py` | `float(torchvision.__version__[:3]) < 0.7` mis-parses `"0.21"` as `0.2 < 0.7` | Tried to import removed `torchvision.ops.misc.interpolate` | Use `packaging.Version` for comparison; replace with `F.interpolate` | During initial setup |
| B6 | `models/ops/functions/ms_deform_attn_func.py` | Compiled `.so` not on `sys.path` when running from `relationformer/` | `ModuleNotFoundError: MultiScaleDeformableAttention` | Add `sys.path.insert` to inject `ops/` dir | During initial setup |
| B7 | `ms_deform_attn_cuda.cu` | Deprecated `value.type()` and `.data<T>()` API removed in PyTorch 2.x | CUDA compilation failure | Patch to `value.scalar_type()` and `.data_ptr<T>()` | During CUDA op build |

**Critical lessons:**
- Always verify R@K with a **perfect-prediction sanity test** before starting a training run.
  Perfect predictions should score R@100 ≈ 0.95 (not exactly 1.0 due to IoU filtering of tiny boxes).
- Augmentation that changes relative box sizes (crop, random scale) must be
  validated against the eval set box size distribution before training.
- Run `scripts/experiments/verify_eval_pipeline.py` before any new training run.

---

## 13. References

1. Stürmer, Graumann, Koch. "From Engineering Diagrams to Graphs: Digitizing P&IDs
   with Transformers." IEEE DSAA 2025. arXiv:2411.13929.
   Dataset: https://zenodo.org/records/14803338

2. Shit et al. "Relationformer: A Unified Framework for Image-to-Graph Generation."
   ECCV 2022. arXiv:2203.10202. Code: https://github.com/suprosanna/relationformer

3. Im et al. "EGTR: Extracting Graph from Transformer for Scene Graph Generation."
   CVPR 2024 (oral, best paper candidate). arXiv:2404.02072.
   Code: https://github.com/naver-ai/egtr

4. Cong, Yang, Rosenhahn. "RelTR: Relation Transformer for Scene Graph Generation."
   IEEE T-PAMI 2023. arXiv:2201.11460. Code: https://github.com/yrcong/RelTR

5. OPEN-100 dataset (Jamieson et al.): The 12 real-world P&ID images used as seeds
   and evaluation data. Located at `data/PID2Graph OPEN100/`.
