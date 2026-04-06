#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master experiment launcher
# =============================================================================
# Runs all experiments (E1–E4) across all 5 folds and 3 seeds sequentially.
# Each experiment is a separate torchrun job. Logs go to outputs/logs/.
#
# Usage:
#   # Run a single experiment across all folds/seeds:
#   bash scripts/experiments/run_all.sh E1
#   bash scripts/experiments/run_all.sh E2
#   bash scripts/experiments/run_all.sh E3
#   bash scripts/experiments/run_all.sh E4
#
#   # Run everything (very long — ~2h per fold per experiment):
#   bash scripts/experiments/run_all.sh all
#
#   # Dry-run (print commands without executing):
#   DRY_RUN=1 bash scripts/experiments/run_all.sh E2
#
# Environment:
#   NGPU=4          Number of GPUs (default 4)
#   BATCH_SIZE=2    Per-GPU batch size (default 2)
#   FOLDS="0 1 2 3 4"  Which folds to run (default all 5)
#   SEEDS="42 123 7"   Which seeds to run (default 3)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RF_DIR="$REPO_ROOT/third_party/relationformer"
CONFIG="$RF_DIR/configs/pid.yaml"
LOG_DIR="$REPO_ROOT/outputs/logs"
RESULTS_DIR="$REPO_ROOT/outputs/experiment_results"
TORCHRUN="/home/pinak/miniconda3/bin/torchrun"

NGPU="${NGPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DRY_RUN="${DRY_RUN:-0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-42 123 7}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
run_one() {
    local experiment="$1"
    local fold="$2"
    local seed="$3"
    local exp_name="${experiment}_fold${fold}_seed${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    log "START  $exp_name"

    local cmd="$TORCHRUN --nproc_per_node=${NGPU} \
        --master_port=$((29500 + fold * 10 + seed % 10)) \
        train_pid.py \
        --config '$CONFIG' \
        --exp-name '$exp_name' \
        --batch-size $BATCH_SIZE \
        --experiment $experiment \
        --fold $fold"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY-RUN: $cmd"
        return
    fi

    PYTHONHASHSEED=$seed eval "$cmd" 2>&1 | tee "$log_file"
    log "DONE   $exp_name  →  $log_file"
}

eval_one() {
    local experiment="$1"
    local fold="$2"
    local seed="$3"
    local exp_name="${experiment}_fold${fold}_seed${seed}"
    local ckpt="$RF_DIR/trained_weights/pid/runs/${exp_name}_${seed}/best_model.pth"
    local out="$RESULTS_DIR/${exp_name}_eval.json"

    if [ ! -f "$ckpt" ]; then
        log "WARN  no checkpoint for $exp_name — skipping eval"
        return
    fi

    log "EVAL   $exp_name"
    local cmd="python eval_pid.py \
        --config '$CONFIG' \
        --checkpoint '$ckpt' \
        --experiment $experiment \
        --fold $fold \
        --output '$out'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY-RUN: $cmd"
        return
    fi

    eval "cd '$RF_DIR' && $cmd" 2>&1 | tee "$LOG_DIR/${exp_name}_eval.log"
    log "SAVED  $out"
}

run_experiment() {
    local experiment="$1"
    log "====== EXPERIMENT: $experiment ======"
    for fold in $FOLDS; do
        for seed in $SEEDS; do
            (cd "$RF_DIR" && run_one "$experiment" "$fold" "$seed")
            eval_one "$experiment" "$fold" "$seed"
        done
    done
    log "====== DONE: $experiment ======"
    python "$REPO_ROOT/scripts/experiments/aggregate_results.py" \
        --results-dir "$RESULTS_DIR" \
        --experiment "$experiment"
}

# ---------------------------------------------------------------------------
MODE="${1:-help}"
case "$MODE" in
    E1) run_experiment "E1_oracle" ;;
    E2) run_experiment "E2_synth_only" ;;
    E3) run_experiment "E3_mixed" ;;
    E4)
        for scale in 40 80 120 165; do
            run_experiment "E4_scale_${scale}"
        done
        ;;
    all)
        run_experiment "E1_oracle"
        run_experiment "E2_synth_only"
        run_experiment "E3_mixed"
        for scale in 40 80 120 165; do
            run_experiment "E4_scale_${scale}"
        done
        ;;
    help|*)
        echo "Usage: bash $0 [E1|E2|E3|E4|all]"
        echo "  E1   oracle (real train, real eval)"
        echo "  E2   synth-only (core claim)"
        echo "  E3   mixed (real + synth)"
        echo "  E4   scale ablation (40/80/120/165 synth)"
        echo "  all  run everything"
        ;;
esac
