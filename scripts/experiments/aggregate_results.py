"""
aggregate_results.py
====================
Aggregate eval JSON files across folds and seeds into a summary table.
Produces:
  - Console table (mean ± std per metric)
  - outputs/experiment_results/<experiment>_summary.json
  - outputs/experiment_results/paper_table.md  (full comparison across experiments)

Usage:
  # Aggregate one experiment:
  python scripts/experiments/aggregate_results.py \\
      --experiment E2_synth_only

  # Regenerate the full paper comparison table:
  python scripts/experiments/aggregate_results.py --paper-table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "experiment_results"

# Metrics to include in the paper table (in display order)
PAPER_METRICS = [
    ("node_mAP50",    "Node mAP@0.5"),
    ("edge_F1_mean",  "Edge F1 (mean)"),
    ("edge_P_mean",   "Edge Precision"),
    ("edge_R_mean",   "Edge Recall"),
    ("edge_F1_solid", "Edge F1 (solid)"),
    ("edge_F1_non-solid", "Edge F1 (non-solid)"),
]

# All experiments in paper order
EXPERIMENTS = [
    "E1_oracle",
    "E2_synth_only",
    "E3_mixed",
    "E4_scale_40",
    "E4_scale_80",
    "E4_scale_120",
    "E4_scale_165",
]

EXPERIMENT_LABELS = {
    "E1_oracle":      "E1: Oracle (real only)",
    "E2_synth_only":  "E2: Synth-only (ours)",
    "E3_mixed":       "E3: Mixed (synth+real)",
    "E4_scale_40":    "E4: Scale-40 synth",
    "E4_scale_80":    "E4: Scale-80 synth",
    "E4_scale_120":   "E4: Scale-120 synth",
    "E4_scale_165":   "E4: Scale-165 synth",
}


def load_eval_files(experiment: str, results_dir: Path) -> list[dict]:
    """Load all eval JSON files for a given experiment."""
    pattern = f"{experiment}_fold*_eval.json"
    files = sorted(results_dir.glob(pattern))
    if not files:
        return []
    results = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
                metrics = d.get("metrics", d)
                results.append(metrics)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    return results


def aggregate(results: list[dict]) -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, std)} across all runs."""
    if not results:
        return {}
    keys = list(results[0].keys())
    return {
        k: (
            float(np.mean([r.get(k, 0.0) for r in results])),
            float(np.std( [r.get(k, 0.0) for r in results])),
        )
        for k in keys
    }


def print_summary(experiment: str, agg: dict[str, tuple[float, float]], n: int) -> None:
    print(f"\n{'='*65}")
    print(f"  {EXPERIMENT_LABELS.get(experiment, experiment)}  ({n} runs)")
    print(f"{'='*65}")
    print(f"  {'Metric':<35}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*55}")
    for k, (mean, std) in sorted(agg.items()):
        print(f"  {k:<35}  {mean:>8.4f}  {std:>8.4f}")
    print(f"{'='*65}")


def save_summary(experiment: str, results: list[dict], agg: dict, results_dir: Path) -> None:
    out = {
        "experiment": experiment,
        "n_runs": len(results),
        "mean": {k: v[0] for k, v in agg.items()},
        "std":  {k: v[1] for k, v in agg.items()},
        "all_runs": results,
    }
    path = results_dir / f"{experiment}_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {path}")


def generate_paper_table(results_dir: Path) -> str:
    """Generate a markdown table comparing all experiments."""
    rows = []
    for exp in EXPERIMENTS:
        summary_path = results_dir / f"{exp}_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        mean = s.get("mean", {})
        std  = s.get("std", {})
        n    = s.get("n_runs", 0)
        rows.append((exp, mean, std, n))

    if not rows:
        return "No results available yet."

    # Header
    col_width = 28
    lines = []
    lines.append("# Experiment Results\n")
    lines.append(f"*Generated from {len(rows)} experiment(s). Values: mean ± std.*\n")

    header = f"| {'Experiment':<30} |"
    sep    = f"| {'-'*30} |"
    for _, label in PAPER_METRICS:
        header += f" {label:^18} |"
        sep    += f" {'-'*18} |"
    lines.append(header)
    lines.append(sep)

    for exp, mean, std, n in rows:
        label = EXPERIMENT_LABELS.get(exp, exp)
        row = f"| {label:<30} |"
        for metric_key, _ in PAPER_METRICS:
            m = mean.get(metric_key, 0.0)
            s = std.get(metric_key, 0.0)
            cell = f"{m:.3f}±{s:.3f}"
            row += f" {cell:^18} |"
        lines.append(row)

    # E4 scale summary — just the edge F1 mean for clarity
    e4_rows = [(exp, mean, std, n) for exp, mean, std, n in rows if exp.startswith("E4")]
    if len(e4_rows) >= 2:
        lines.append("\n## E4 Scale Ablation (Edge F1 mean)\n")
        lines.append("| Scale | n_synth | Edge F1 (mean) |")
        lines.append("| ----- | ------- | -------------- |")
        for exp, mean, std, n in e4_rows:
            scale = exp.replace("E4_scale_", "")
            f1 = mean.get("edge_F1_mean", 0.0)
            f1s = std.get("edge_F1_mean", 0.0)
            lines.append(f"| {scale:>5} | {scale:>7} | {f1:.3f}±{f1s:.3f} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P&ID experiment results")
    parser.add_argument("--experiment", default=None,
                        help="Single experiment to aggregate (e.g. E2_synth_only)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--paper-table", action="store_true",
                        help="Regenerate the full paper comparison table from all summaries")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_table or args.experiment is None:
        # Aggregate all experiments that have results
        for exp in EXPERIMENTS:
            results = load_eval_files(exp, args.results_dir)
            if not results:
                continue
            agg = aggregate(results)
            print_summary(exp, agg, len(results))
            save_summary(exp, results, agg, args.results_dir)

        table = generate_paper_table(args.results_dir)
        table_path = args.results_dir / "paper_table.md"
        table_path.write_text(table)
        print(f"\nPaper table → {table_path}")
        print("\n" + table)

    else:
        results = load_eval_files(args.experiment, args.results_dir)
        if not results:
            print(f"No eval files found for '{args.experiment}' in {args.results_dir}")
            return
        agg = aggregate(results)
        print_summary(args.experiment, agg, len(results))
        save_summary(args.experiment, results, agg, args.results_dir)


if __name__ == "__main__":
    main()
