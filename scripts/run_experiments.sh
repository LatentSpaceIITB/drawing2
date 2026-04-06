#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# =============================================================================
# Orchestrates all 5 experiments (E1–E4 + baseline) across 5 CV folds and 3
# random seeds, as defined in docs/research_plan.md.
#
# Prerequisites (run once before this script):
#   cd /path/to/drawing2
#   python scripts/preprocess/collapse_connectors.py \
#       --input-dir "data/PID2Graph OPEN100" \
#       --output-dir outputs/preprocessed/real_collapsed
#   python scripts/preprocess/collapse_connectors.py \
#       --input-dir outputs/open100_dense_1000_v2 \
#       --output-dir outputs/preprocessed/synth_collapsed
#   python scripts/preprocess/graphml_to_coco.py \
#       --graphml-dir outputs/preprocessed/real_collapsed \
#       --image-dir "data/PID2Graph OPEN100" \
#       --output outputs/preprocessed/real_coco.json
#   python scripts/preprocess/graphml_to_coco.py \
#       --graphml-dir outputs/preprocessed/synth_collapsed \
#       --image-dir outputs/open100_dense_1000_v2 \
#       --output outputs/preprocessed/synthetic_coco.json
#   python scripts/preprocess/split_dataset.py \
#       --real-dir "data/PID2Graph OPEN100" \
#       --synth-dir outputs/open100_dense_1000_v2 \
#       --output outputs/preprocessed/splits.json
#
# Usage:
#   # Run a single experiment (useful for initial testing):
#   EXPERIMENT=E2_synth_only FOLD=0 SEED=42 bash scripts/run_experiments.sh single
#
#   # Run all experiments (long):
#   bash scripts/run_experiments.sh all
#
#   # Run only E2 across all folds and seeds:
#   bash scripts/run_experiments.sh E2
#
# Environment variables:
#   NGPU        Number of GPUs to use (default: 4)
#   BATCH_SIZE  Per-GPU batch size (default: 2)
#   EPOCHS      Override training epochs (default: from config)
#   DRY_RUN     Set to 1 to print commands without executing
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (edit if your layout differs)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RF_DIR="$REPO_ROOT/third_party/relationformer"
CONFIG="$RF_DIR/configs/pid.yaml"
RESULTS_DIR="$REPO_ROOT/outputs/experiment_results"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NGPU="${NGPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS=(42 123 7)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_cmd() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY-RUN: $*"
    else
        eval "$@"
    fi
}

train_one() {
    local experiment="$1"
    local fold="$2"
    local seed="$3"
    local exp_name="${experiment}_fold${fold}_seed${seed}"

    log "=== TRAIN $exp_name ==="

    local cmd="/home/pinak/miniconda3/bin/torchrun \
        --nproc_per_node=${NGPU} \
        --master_port=$((29500 + fold * 10 + seed % 10)) \
        '$RF_DIR/train_pid.py' \
        --config '$CONFIG' \
        --exp-name '$exp_name' \
        --batch-size $BATCH_SIZE \
        --experiment $experiment \
        --fold $fold"

    # Override seed via env if the config supports it
    export PYTHONHASHSEED=$seed

    run_cmd "cd '$RF_DIR' && $cmd" 2>&1 | tee "$RESULTS_DIR/${exp_name}_train.log"
    log "=== DONE TRAIN $exp_name ==="
}

eval_one() {
    local experiment="$1"
    local fold="$2"
    local seed="$3"
    local exp_name="${experiment}_fold${fold}_seed${seed}"
    local ckpt_dir="$RF_DIR/trained_weights/pid/runs/${exp_name}_${seed}"
    local best_ckpt="$ckpt_dir/best_model.pth"

    if [ ! -f "$best_ckpt" ]; then
        log "WARNING: No checkpoint found at $best_ckpt — skipping eval"
        return
    fi

    log "=== EVAL $exp_name ==="
    local out_json="$RESULTS_DIR/${exp_name}_eval.json"

    local cmd="python '$RF_DIR/eval_pid.py' \
        --config '$CONFIG' \
        --checkpoint '$best_ckpt' \
        --experiment $experiment \
        --fold $fold \
        --output '$out_json'"

    run_cmd "cd '$RF_DIR' && $cmd" 2>&1 | tee "$RESULTS_DIR/${exp_name}_eval.log"
    log "=== DONE EVAL $exp_name → $out_json ==="
}

run_experiment() {
    local experiment="$1"
    log "############################################################"
    log "# Experiment: $experiment"
    log "# Folds: 0-4    Seeds: ${SEEDS[*]}"
    log "############################################################"
    for fold in 0 1 2 3 4; do
        for seed in "${SEEDS[@]}"; do
            train_one "$experiment" "$fold" "$seed"
            eval_one  "$experiment" "$fold" "$seed"
        done
    done
}

aggregate_results() {
    local experiment="$1"
    log "=== AGGREGATE $experiment ==="
    python3 - <<PYEOF
import json, glob, numpy as np, sys

pattern = "$RESULTS_DIR/${experiment}_fold*_seed*_eval.json"
files = sorted(glob.glob(pattern))
if not files:
    print(f"No result files found for pattern: {pattern}")
    sys.exit(0)

all_metrics = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
        all_metrics.append(d.get("metrics", d))

keys = list(all_metrics[0].keys())
print(f"\n{'Metric':<35} {'Mean':>8}  {'Std':>8}  (n={len(all_metrics)})")
print("-" * 60)
for k in keys:
    vals = [m.get(k, 0.0) for m in all_metrics]
    print(f"  {k:<33} {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")

summary = {
    "experiment": "$experiment",
    "n_runs":     len(all_metrics),
    "mean":       {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
    "std":        {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
}
out = "$RESULTS_DIR/${experiment}_summary.json"
with open(out, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nSummary saved → {out}")
PYEOF
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

mkdir -p "$RESULTS_DIR"

MODE="${1:-all}"

case "$MODE" in
    single)
        # Single run: use env vars EXPERIMENT FOLD SEED
        EXP="${EXPERIMENT:-E2_synth_only}"
        FOLD="${FOLD:-0}"
        SEED_VAL="${SEED:-42}"
        train_one "$EXP" "$FOLD" "$SEED_VAL"
        eval_one  "$EXP" "$FOLD" "$SEED_VAL"
        ;;

    E1)
        run_experiment "E1_oracle"
        aggregate_results "E1_oracle"
        ;;

    E2)
        run_experiment "E2_synth_only"
        aggregate_results "E2_synth_only"
        ;;

    E3)
        run_experiment "E3_mixed"
        aggregate_results "E3_mixed"
        ;;

    E4)
        for scale in 40 80 120 165; do
            run_experiment "E4_scale_${scale}"
            aggregate_results "E4_scale_${scale}"
        done
        ;;

    all)
        log "Running ALL experiments (E1–E4). This will take a long time."
        log "Tip: run individual experiments with: bash $0 E2"

        run_experiment "E1_oracle"
        aggregate_results "E1_oracle"

        run_experiment "E2_synth_only"
        aggregate_results "E2_synth_only"

        run_experiment "E3_mixed"
        aggregate_results "E3_mixed"

        for scale in 40 80 120 165; do
            run_experiment "E4_scale_${scale}"
            aggregate_results "E4_scale_${scale}"
        done

        log "All experiments complete. Results in $RESULTS_DIR"
        ;;

    *)
        echo "Usage: $0 [single|E1|E2|E3|E4|all]"
        echo "  single  — run EXPERIMENT/FOLD/SEED from env vars"
        echo "  E1      — oracle (real train, real eval)"
        echo "  E2      — synth-only (core claim)"
        echo "  E3      — mixed (real + synth train)"
        echo "  E4      — scale ablation (40/80/120/165 synth)"
        echo "  all     — run everything"
        exit 1
        ;;
esac
