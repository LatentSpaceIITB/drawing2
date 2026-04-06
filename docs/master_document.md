# SynthPID: Master Project Document

**Project:** Synthetic-to-Real Transfer for P&ID Graph Extraction  
**Last updated:** 2026-03-16  
**Status:** Training in progress (E2, fold 0, epoch ~5/50)

This is the single authoritative document for the project. Read this first.
All other docs in this folder are supplementary detail.

---

## Table of Contents

1. [The One-Paragraph Summary](#1-the-one-paragraph-summary)
2. [Problem Statement](#2-problem-statement)
3. [Selling Points and Paper Contribution](#3-selling-points-and-paper-contribution)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Model Architecture](#6-model-architecture)
7. [Experiments and Benchmarking](#7-experiments-and-benchmarking)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Implementation Status](#9-implementation-status)
10. [Known Issues and Decisions](#10-known-issues-and-decisions)
11. [Training Observations](#11-training-observations)
12. [Next Steps](#12-next-steps)
13. [File Structure](#13-file-structure)
14. [References](#14-references)

---

## 1. The One-Paragraph Summary

P&ID (Piping and Instrumentation Diagram) digitization — converting raster
engineering drawings into structured process graphs — is critical for the oil &
gas, chemical, and power industries but blocked by data scarcity: no company
shares proprietary plant drawings. We address this by generating synthetic P&IDs
from a small real-world seed set, and demonstrating that a graph-extraction
transformer (Relationformer) trained **only on synthetic data** generalises to
unseen real P&IDs. Using the only public annotated P&ID benchmark (OPEN-100, 12
images), we show this synthetic-to-real transfer works, establish performance
benchmarks, and release the synthetic dataset and generation pipeline. Target
venue: IEEE/CVF computer vision or industrial AI conference (CVPR, ICCV, ICASSP,
or domain-specific: DSAA, ECML).

---

## 2. Problem Statement

### Formal task

Given a raster P&ID image `I`, output the process graph `G = (V, E)` where:

- **V** = set of detected physical components, each with bounding box + class label
  from {`valve`, `pump`, `instrumentation`, `general`, `tank`, `arrow`, `inlet/outlet`}
- **E** = set of pipe/signal connections between components, each with an edge label
  from {`solid`, `non-solid`}

### What is NOT in scope

- OCR / instrument tag recognition
- Pixel-level pipe tracing
- Multi-sheet or cross-reference resolution

### Pre-processing decision: connector collapsing

Raw GraphML annotations contain `connector` nodes (pipe routing waypoints, ~8×8 px)
and `crossing` nodes (pipe-over-pipe non-connections). These make up 50–65% of all
nodes but carry no physical meaning. We remove them and connect the physical
components they bridge (majority-vote edge label). See `research_plan.md §3` for
the full algorithm and trade-off discussion.

**After collapsing:** a typical real image has ~60–210 physical nodes and ~28–118
edges. A typical synthetic image has ~150–280 physical nodes and ~100–200 edges.

---

## 3. Selling Points and Paper Contribution

### Primary selling point

> **You do not need proprietary plant data to digitize P&IDs.**
> A handful of public reference diagrams is enough to generate a training corpus
> that transfers to unseen real-world drawings.

This is directly relevant to every engineering company that wants to automate
P&ID digitization but cannot share their drawings with an ML vendor.

### Specific contributions

1. **Synthetic-to-real transfer baseline** — first systematic study of whether
   synthetic P&ID training data generalises to real drawings, using a rigorous
   evaluation protocol (5-fold CV, 3 seeds, mean ± std).

2. **SynthPID dataset** — 165 synthetic P&ID images with full graph annotations
   (bounding boxes + class labels + edges), generated from 12 real seeds.
   Released under open license. Larger than any existing annotated P&ID dataset.

3. **Generation pipeline** — a reproducible pipeline (Phase 1) that creates
   structured synthetic P&IDs with WL-hash and perceptual-hash deduplication,
   ensuring graph and visual diversity. Can be scaled to thousands of images.

4. **Benchmarking on OPEN-100** — we use the only public P&ID graph benchmark
   (Stürmer et al., 2025) as our real-world evaluation set, providing results
   directly comparable to the state of the art.

5. **Scaling story** — an ablation showing how performance varies with synthetic
   training set size (40 → 80 → 120 → 165 samples), informing how much synthetic
   data is needed.

### Why this is publishable

- The data scarcity argument is universally understood and acknowledged in prior
  work (Stürmer et al. explicitly call it out as a limitation).
- Synthetic-to-real transfer is a well-studied problem in vision but has not been
  applied to engineering diagram digitization.
- We have both the synthetic generation pipeline AND the evaluation infrastructure
  ready — this is not a speculative contribution.
- Our evaluation dataset (OPEN-100) is identical to the SOTA paper, so results
  are directly comparable without re-running baselines.

### What would make this stronger

- R@K numbers on real data that are non-trivially above 0 (expected after epoch 30)
- Showing a clear improvement over the Stürmer et al. modular baseline
- A qualitative figure showing predicted graphs overlaid on real P&IDs
- The scaling curve (E4) showing monotonic improvement with synthetic data volume

---

## 4. Dataset

### Real-world data (evaluation only)

| Property | Value |
|---|---|
| Name | OPEN-100 |
| Source | Stürmer et al., IEEE DSAA 2025 (arXiv:2411.13929) |
| Location | `data/PID2Graph OPEN100/` |
| Size | 12 images (paired PNG + GraphML) |
| Image size | ~2604×1744 px (variable) |
| Nodes (after collapse) | 57–210 physical components per image |
| Edges (after collapse) | 28–118 connections per image |
| Node classes | valve, pump, instrumentation, general, tank, arrow, inlet/outlet |
| Edge classes | solid (dominant), non-solid |
| Use | Evaluation only — 5-fold CV |

### Synthetic data (training only)

| Property | Value |
|---|---|
| Name | SynthPID-165 |
| Source | Our Phase-1 generation pipeline |
| Location | `outputs/open100_dense_1000_v2/` |
| Size | 165 images (PNG + GraphML + SVG + manifest) |
| Image size | 7168×4562 px (fixed) |
| Nodes (after collapse) | 150–280 physical components per image |
| Edges (after collapse) | 100–200 connections per image |
| Generation | 12,000+ attempts; 165 accepted after WL-hash + perceptual-hash dedup |
| Use | Training only |

### Preprocessed data (generated, not in git)

Located in `outputs/preprocessed/`:
- `real_collapsed/` — 12 GraphML files with connectors removed
- `synth_collapsed/` — 165 GraphML files with connectors removed
- `real_coco.json` — COCO-format annotation for real images (absolute paths)
- `synthetic_coco.json` — COCO-format annotation for synthetic images
- `splits.json` — 5-fold CV split indices covering all 5 experiment configs

---

## 5. Methodology

### Pipeline overview

```
Real P&ID images (12)
        │
        ├──► Preprocess ──────────────────────────────► COCO JSON (real)
        │    collapse_connectors.py                            │
        │    graphml_to_coco.py                                │
        │                                                      ▼
        └──► Phase-1 Generator ──► 165 Synthetic ──► COCO JSON (synth)
             (open100_structural_generation)                   │
                                                               ▼
                                                      split_dataset.py
                                                               │
                                                               ▼
                                                       5-fold CV splits
                                                               │
                                                               ▼
                                                      Relationformer
                                                      (train_pid.py)
                                                               │
                                                               ▼
                                                      eval_pid.py
                                                               │
                                                               ▼
                                                      aggregate_results.py
                                                               │
                                                               ▼
                                                      paper_table.md
```

### Phase 1: Synthetic data generation

The generation pipeline (documented in `phase1_architecture.md` and
`open100_structural_generation.md`) takes a real GraphML as a structural seed and
generates new P&IDs by:
1. Sampling a subset of the seed graph's topology
2. Applying spatial perturbations and symbol variations
3. Rendering to PNG + SVG + GraphML
4. Deduplicating via WL-hash (graph structure) + perceptual hash (visual)

This ensures generated images are structurally diverse and visually distinct from
each other and from the seed images.

### Phase 2: ML experiment

The core experimental loop:
1. **Preprocess**: collapse connectors, convert to COCO JSON, generate CV splits
2. **Train**: Relationformer on synthetic images (or real, or mixed)
3. **Evaluate**: on held-out real images; R@K and node mAP
4. **Aggregate**: mean ± std across 5 folds × 3 seeds
5. **Analyse**: per-class errors, qualitative overlays, scaling curve

---

## 6. Model Architecture

### Primary: Relationformer (ECCV 2022)

**Paper:** Shit et al., "Relationformer: A Unified Framework for Image-to-Graph
Generation." arXiv:2203.10202.

**Why chosen:**
- Stürmer et al. used it on OPEN-100 → direct numerical comparison
- Joint detection + relation prediction in a single forward pass
- Built on Deformable-DETR → ImageNet pretrained backbone available
- One relation token (rln-token) shared across all pairs → undirected edges, which
  matches P&ID connectivity

**Architecture summary:**
- Backbone: ResNet-50 (ImageNet pretrained)
- Encoder: Deformable Transformer encoder (6 layers, 256-dim hidden)
- Decoder: 300 object queries + 1 relation token
- Node head: FFN → class logits (8 classes: bg + 7 physical) + bbox (4 coords)
- Edge head: rln-token ⊗ object-token pairs → 3-class CE (0=no-edge, 1=solid, 2=non-solid)

**Config:** `third_party/relationformer/configs/pid.yaml`

**Key hyperparameters:**
```yaml
NUM_OBJ_CLS:  7       # physical classes (bg excluded from count)
NUM_REL_CLS:  2       # solid, non-solid
NUM_QUERIES:  300     # > max 280 physical nodes per synthetic image
HIDDEN_DIM:   256
EPOCHS:       50
LR:           1e-4    # drops to 1e-5 at epoch 40
BATCH_SIZE:   2       # per GPU; effective batch 8 (4x A6000)
W_EDGE:       5.0     # edge loss weight (higher than VG default of 3.0)
```

### Comparison baselines

| Model | Role |
|---|---|
| Modular pipeline (detect + trace) | Classical baseline from Stürmer et al. |
| Relationformer (Stürmer et al. numbers) | Published SOTA on OPEN-100 |
| EGTR (Im et al., CVPR 2024) | Modern SGG model; secondary comparison |

---

## 7. Experiments and Benchmarking

### Experiment table

| ID | Name | Train | Eval | Purpose | Status |
|---|---|---|---|---|---|
| **E1** | Oracle | 12 real (CV train fold) | 12 real (CV val fold) | Upper bound — best possible with real data | Not started |
| **E2** | Synth-only | 165 synthetic | 12 real (all folds) | **Core claim** — does synth→real transfer work? | **Running** (fold 0, epoch ~5) |
| **E3** | Mixed | 165 synth + real train fold | 12 real (val fold) | Practical best-case: few real + synthetic | Not started |
| **E4** | Scale ablation | 40/80/120/165 synth | 12 real (all folds) | How much synthetic data is needed? | Not started |
| E5 | Baseline replication | Stürmer et al. setup | OPEN-100 | Validate eval matches prior work | Not started |

### Expected results narrative

The paper's story told through experiments:

1. **E2 (synth-only) > 0** → synthetic training transfers at all; core claim holds
2. **E1 (oracle) > E2** → real data is still better; expected and honest
3. **E3 (mixed) ≈ E1** → adding a little real data closes the gap quickly
4. **E4 (scale curve) monotone** → more synthetic = better; motivates larger corpora
5. **E2 vs Stürmer baseline** → if E2 beats the modular baseline, very strong result

### Benchmarking against prior work

The paper comparison table (target):

| Method | Train data | Node mAP@0.5 | Edge F1 | Edge R@100 |
|---|---|---|---|---|
| Modular pipeline [Stürmer 2025] | 12 real | TBD | TBD | TBD |
| Relationformer [Stürmer 2025] | 12 real | TBD | TBD | TBD |
| **Ours — E2 (synth-only)** | 165 synth | ? | ? | ? |
| **Ours — E1 (oracle)** | 12 real | ? | ? | ? |
| **Ours — E3 (mixed)** | 165 synth + real | ? | ? | ? |

### Cross-validation and statistical reporting

- 5 folds × 3 seeds = 15 runs per experiment
- Report: **mean ± std** for all metrics
- Small N (3 val images per fold) means high variance — std must be reported
- Folds are fixed (seed=42, stored in `outputs/preprocessed/splits.json`)

### Commands to run experiments

```bash
cd /home/pinak/MedClip/drawing2/third_party/relationformer

# E2 (currently running — fold 0):
/home/pinak/miniconda3/bin/torchrun --nproc_per_node=4 train_pid.py \
    --config configs/pid.yaml --exp-name E2_synth_only_fold0 \
    --batch-size 2 --experiment E2_synth_only --fold 0

# After fold 0 completes, run all remaining experiments:
cd /home/pinak/MedClip/drawing2
bash scripts/experiments/run_all.sh E2    # all 5 folds, 3 seeds
bash scripts/experiments/run_all.sh E1
bash scripts/experiments/run_all.sh E3
bash scripts/experiments/run_all.sh E4

# Generate paper results table at any time:
python scripts/experiments/aggregate_results.py --paper-table
```

---

## 8. Evaluation Metrics

### Primary metrics (reported in paper)

**Node detection — mAP@0.5 IoU**
- A predicted node is correct if IoU(pred_box, gt_box) ≥ 0.5 AND class matches
- Computed per class + mean across 7 classes
- Implementation: `eval_pid.py::NodeEdgeMetrics`

**Edge detection — Recall@K (R@20, R@50, R@100)**
- Standard scene graph generation metric
- A predicted edge is correct if:
  - Both endpoint nodes are correctly detected (IoU≥0.5, correct class)
  - The edge label (solid/non-solid) matches
- Recall@K = fraction of GT edges correctly predicted in the top-K predictions
- Implementation: `util/sg_recall.py::BasicSceneGraphEvaluator`

**Edge F1 (overall + per predicate)**
- Precision, Recall, F1 on (node_i, edge_label, node_j) triplets
- Reported separately for `solid` and `non-solid`
- Implementation: `eval_pid.py::NodeEdgeMetrics`

### Supplementary metric

**Graph Edit Distance (GED)**
- Measures structural similarity of full predicted graph vs GT
- NP-hard for large graphs; approximated
- Not used for ranking models; reported as supplementary

### Critical implementation notes

The evaluation pipeline has been carefully validated:

1. GT boxes are in normalised cxcywh after transforms — pass directly to evaluator
   (do NOT re-convert; the evaluator calls `box_cxcywh_to_xyxy` internally)
2. Predicted labels from `graph_infer` are 0-indexed — pass as-is to evaluator
   (GT labels must also be 0-indexed: use `gt["labels"] - 1`)
3. Perfect predictions score R@100 ≈ 0.95 (not 1.0 due to IoU filtering of tiny
   boxes with `use_gt_filter=True`)
4. Run `python scripts/experiments/verify_eval_pipeline.py` before any new training
   run to catch regressions

### When to expect non-zero R@K

DETR-based models need many epochs before box regression converges:
- Epoch 5: R@K=0 expected (boxes 7–11× too large, only 2/210 GT matched at IoU≥0.5)
- Epoch 20–30: first non-zero R@K values expected
- Epoch 50: end of training; meaningful results for paper

---

## 9. Implementation Status

### What is complete

| Component | Location | Status |
|---|---|---|
| Phase-1 synthetic generator | `src/`, `generator/` | Complete |
| Connector collapse script | `scripts/preprocess/collapse_connectors.py` | Complete, tested |
| GraphML→COCO converter | `scripts/preprocess/graphml_to_coco.py` | Complete, tested |
| CV split generator | `scripts/preprocess/split_dataset.py` | Complete, tested |
| Preprocessed data | `outputs/preprocessed/` | Generated |
| Relationformer clone | `third_party/relationformer/` | Complete (scene_graph branch) |
| P&ID data loader | `third_party/relationformer/dataset_pid.py` | Complete, tested |
| P&ID config | `third_party/relationformer/configs/pid.yaml` | Complete |
| Training script | `third_party/relationformer/train_pid.py` | Complete, tested (4×A6000 DDP) |
| Evaluation script | `third_party/relationformer/eval_pid.py` | Complete |
| Experiment launcher | `scripts/experiments/run_all.sh` | Complete |
| Results aggregator | `scripts/experiments/aggregate_results.py` | Complete |
| Visualisation script | `scripts/experiments/visualise_predictions.py` | Complete |
| Pre-flight check | `scripts/experiments/verify_eval_pipeline.py` | Complete, all 4 checks pass |

### What is in progress

| Component | Status |
|---|---|
| E2 training (fold 0, seed 42) | Epoch ~5/50, loss=47.1 and decreasing |

### What is not yet started

| Component | Notes |
|---|---|
| E1, E3, E4 experiments | Launch after E2 fold 0 completes |
| All remaining E2 folds (1–4) | Part of `run_all.sh E2` |
| Paper writing | Start after first full experiment set |
| Qualitative figures | `visualise_predictions.py` ready; needs trained checkpoint |
| Comparison with Stürmer et al. baseline | Need to replicate their numbers (E5) |

---

## 10. Known Issues and Decisions

### Bugs found and fixed (complete log)

| # | Bug | Impact | Fix |
|---|---|---|---|
| B1 | `RandomSizeCrop` in training augmentation | Predicted boxes 9.5× too large → R@K=0 for 50 epochs | Removed; use `RandomResize` only |
| B2 | GT boxes double-converted to cxcywh in `evaluate()` | Evaluator received scrambled coordinates | Pass boxes directly (already cxcywh after `Normalize` transform) |
| B3 | `pred_class + 1` in `evaluate()` | Pred labels 1-indexed vs GT 0-indexed → no triplet ever matched | Remove `+1` |
| B4 | `F.cross_entropy` inside `if len(freq_dist) > 0` | Edge loss always `None` when `FREQ_BIAS=False` → crash | Move CE outside condition |
| B5 | `torchvision.__version__` string comparison in `misc.py` | Wrong branch taken → import of removed API | Use `packaging.Version` |
| B6 | CUDA `.so` not on `sys.path` | `ModuleNotFoundError` for deformable attention | Add `sys.path.insert` in func file |
| B7 | Deprecated `value.type()` in CUDA kernel | Compilation failure with PyTorch 2.x | Patch to `value.scalar_type()` |

### Resolved architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Connector collapsing | Yes — collapse before training/eval | Industrial meaning; evaluation cleanliness; class balance |
| Edge label inheritance | Majority vote along chain; ties → solid | Unambiguous rule |
| Crossing nodes | Remove entirely (no edges through them) | Crossings are non-connections |
| Primary model | Relationformer | Direct comparability to Stürmer et al. |
| Evaluation protocol | 5-fold CV, 3 seeds, mean ± std | Standard for small-dataset papers |
| Synthetic corpus | 165 samples from open100_dense_1000_v2 | Highest-quality deduplicated batch |
| Scaling decision | Start at 165; ablate in E4 | Let data tell us if more is needed |

### Open questions

| # | Question | Priority |
|---|---|---|
| Q1 | Will box regression converge by epoch 50 given tiny box sizes (~0.008 normalised)? | High |
| Q2 | Should we pre-train on COCO-pretrained DETR weights to speed convergence? | Medium |
| Q3 | Does OCR of instrument tags help node classification? | Low (out of scope v1) |
| Q4 | Can we generate more synthetic data (500+) if E4 shows scaling benefit? | Medium |
| Q5 | Is IoU@0.5 the right threshold for tiny ~8px symbols? Consider IoU@0.3 | Medium |

---

## 11. Training Observations

### Current run: E2 synth-only fold 0 (v2 — with all fixes)

```
Run name:  E2_synth_only_fold0_v2 (seed 42)
Config:    third_party/relationformer/trained_weights/pid/runs/E2_synth_only_fold0_v2_42/
Hardware:  4× NVIDIA A6000 (48GB), DDP
Batch:     2 per GPU (effective 8)
Epochs:    50 (LR drops at epoch 40: 1e-4 → 1e-5)
~Time/epoch: 2.5 min
```

Loss progression observed:

| Epoch | Total loss | cls | box | edge | R@K |
|---|---|---|---|---|---|
| 1 | 67.46 | 1.75 | 1.23 | 0.82 | 0.0 |
| 2 | 61.50 | 1.60 | 1.11 | 0.77 | 0.0 |
| 3 | 56.40 | 1.39 | 1.10 | 0.74 | 0.0 |
| 4 | 52.44 | 1.03 | 1.18 | 0.65 | 0.0 |
| 5 | 47.05 | 0.73 | 1.18 | 0.53 | 0.0 (val) |

**R@K=0 at epoch 5 is expected** — boxes are 7–11× too large at this stage. The
classification loss is dropping quickly (1.75→0.73), which is good. Box loss is
stuck at ~1.18 — this is the main bottleneck. Expect box loss to start dropping
after epoch 15–20 as the Hungarian matcher stabilises.

### Previous run: E2 synth-only fold 0 (v1 — with augmentation bug)

This run trained 50 epochs but had `RandomSizeCrop` in the augmentation, making
predicted boxes 9.5× too large. It also had the eval coordinate bug and class
index bug. **All results from this run are invalid.** Checkpoints exist at
`trained_weights/pid/runs/E2_synth_only_fold0_42/` but should not be used.

---

## 12. Next Steps

### Immediate (while E2 fold 0 runs)

1. Monitor epoch 20 and 30 checkpoints — first non-zero R@K expected around epoch 25
2. If box loss doesn't start dropping by epoch 15, consider loading COCO-pretrained
   DETR backbone weights (set `MODEL.PRETRAIN` in config)

### After E2 fold 0 completes (~2 hours)

```bash
# Run all 5 folds of E2 with 3 seeds each:
bash scripts/experiments/run_all.sh E2

# Immediately followed by oracle experiment:
bash scripts/experiments/run_all.sh E1
```

### After E1 and E2 complete

1. Run E3 (mixed) and E4 (scale ablation)
2. Generate qualitative figures:
   ```bash
   python scripts/experiments/visualise_predictions.py \
       --checkpoint <best_model.pth> \
       --experiment E2_synth_only --fold 0 \
       --out-dir outputs/visualisations/E2_fold0
   ```
3. Generate the paper results table:
   ```bash
   python scripts/experiments/aggregate_results.py --paper-table
   ```

### Paper writing order

1. Experiments section (fill in numbers as they come)
2. Method section (preprocessing + architecture + training details)
3. Related work (Stürmer et al., DETR, synthetic-to-real transfer lit)
4. Introduction (frame the problem, state contribution, summarise results)
5. Abstract (last)

---

## 13. File Structure

```
drawing2/
│
├── data/
│   └── PID2Graph OPEN100/          # 12 real P&ID images + GraphML
│       ├── 0.png ... 11.png
│       └── 0.graphml ... 11.graphml
│
├── outputs/
│   ├── open100_dense_1000_v2/      # 165 synthetic images (training data)
│   │   ├── open100_dense_00000.png
│   │   ├── open100_dense_00000.graphml
│   │   └── ...
│   ├── preprocessed/               # generated — not in git
│   │   ├── real_collapsed/         # 12 collapsed GraphML files
│   │   ├── synth_collapsed/        # 165 collapsed GraphML files
│   │   ├── real_coco.json          # COCO annotation for real images
│   │   ├── synthetic_coco.json     # COCO annotation for synthetic images
│   │   └── splits.json             # 5-fold CV split indices
│   ├── experiment_results/         # eval JSON files + paper_table.md
│   ├── visualisations/             # qualitative overlay figures
│   └── logs/                       # training logs
│
├── docs/
│   ├── master_document.md          # THIS FILE — read first
│   ├── research_plan.md            # Detailed ML plan (decisions, bugs, refs)
│   ├── roadmap.md                  # Phase 1 done + Phase 2 summary
│   ├── phase1_architecture.md      # Synthetic generator architecture
│   ├── open100_structural_generation.md  # Generator design choices
│   ├── dataset_analysis.md         # Analysis of generation dataset
│   ├── project_journal.md          # Phase 1 development log
│   └── reproducibility.md          # How to reproduce Phase 1 outputs
│
├── scripts/
│   ├── preprocess/
│   │   ├── collapse_connectors.py  # Remove connector/crossing nodes from GraphML
│   │   ├── graphml_to_coco.py      # Convert collapsed GraphML to COCO JSON
│   │   └── split_dataset.py        # Generate 5-fold CV splits
│   └── experiments/
│       ├── run_all.sh              # Master experiment launcher (E1–E4)
│       ├── aggregate_results.py    # Aggregate eval JSONs → paper table
│       ├── visualise_predictions.py # Overlay pred graph on P&ID images
│       └── verify_eval_pipeline.py  # Pre-flight check (run before training)
│
└── third_party/
    └── relationformer/             # Cloned (scene_graph branch) + our additions
        ├── dataset_pid.py          # P&ID-specific data loader
        ├── train_pid.py            # Training entry point (DDP, 4×A6000)
        ├── eval_pid.py             # Standalone evaluation script
        └── configs/
            └── pid.yaml            # P&ID-specific hyperparameters
```

---

## 14. References

1. **Stürmer, Graumann, Koch.** "From Engineering Diagrams to Graphs: Digitizing
   P&IDs with Transformers." IEEE DSAA 2025. arXiv:2411.13929.
   Dataset: https://zenodo.org/records/14803338
   *→ Our baseline, our evaluation dataset, our primary comparison.*

2. **Shit et al.** "Relationformer: A Unified Framework for Image-to-Graph
   Generation." ECCV 2022. arXiv:2203.10202.
   Code: https://github.com/suprosanna/relationformer
   *→ Our primary model.*

3. **Im et al.** "EGTR: Extracting Graph from Transformer for Scene Graph
   Generation." CVPR 2024. arXiv:2404.02072.
   Code: https://github.com/naver-ai/egtr
   *→ Alternative model considered; may be added as secondary comparison.*

4. **Cong, Yang, Rosenhahn.** "RelTR: Relation Transformer for Scene Graph
   Generation." IEEE T-PAMI 2023. arXiv:2201.11460.
   *→ Considered but not chosen (triplet-centric decoder misses isolated nodes).*

5. **Carion et al.** "End-to-End Object Detection with Transformers." ECCV 2020.
   arXiv:2005.12872. *→ Foundation: DETR, which Relationformer is built on.*

6. **Zhu et al.** "Deformable DETR: Deformable Transformers for End-to-End Object
   Detection." ICLR 2021. arXiv:2010.04159.
   *→ Deformable attention used in Relationformer encoder/decoder.*

7. **OPEN-100 P&ID Dataset.** Jamieson et al.
   https://www.open-100.com / https://zenodo.org/records/14803338
   *→ Our 12 real-world images. The only public P&ID graph benchmark.*
