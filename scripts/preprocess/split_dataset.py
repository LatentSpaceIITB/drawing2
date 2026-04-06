"""
split_dataset.py
================
Generate reproducible 5-fold cross-validation split indices for the 12 real
OPEN-100 P&ID images, and optionally combine with synthetic training data.

What this produces
------------------
A JSON file `splits.json` with this structure:

{
  "description": "...",
  "n_folds": 5,
  "real_images": ["0", "1", ..., "11"],
  "synthetic_images": ["open100_dense_00000", ...],   // if provided
  "folds": [
    {
      "fold": 0,
      "real_val":   ["2", "7"],          // stems of real images held out for eval
      "real_train": ["0","1","3",...],    // real images available for training
      "synth_train": ["open100_dense_00000", ...],  // all synthetic (never in val)
      // Experiment configurations for this fold:
      "experiments": {
        "E1_oracle":     {"train": real_train,   "val": real_val},
        "E2_synth_only": {"train": synth_train,  "val": real_val},
        "E3_mixed":      {"train": real_train + synth_train, "val": real_val},
        "E4_scale": {
          "40":  {"train": synth_train[:40],  "val": real_val},
          "80":  {"train": synth_train[:80],  "val": real_val},
          "120": {"train": synth_train[:120], "val": real_val},
          "165": {"train": synth_train[:165], "val": real_val}
        }
      }
    },
    ...
  ]
}

Usage
-----
  # Real data only (5-fold)
  python scripts/preprocess/split_dataset.py \\
      --real-dir "data/PID2Graph OPEN100" \\
      --output   outputs/preprocessed/splits.json

  # Real + synthetic
  python scripts/preprocess/split_dataset.py \\
      --real-dir   "data/PID2Graph OPEN100" \\
      --synth-dir  outputs/open100_dense_1000_v2 \\
      --output     outputs/preprocessed/splits.json

  # Custom seed and fold count
  python scripts/preprocess/split_dataset.py \\
      --real-dir "data/PID2Graph OPEN100" \\
      --n-folds  5 \\
      --seed     42 \\
      --output   outputs/preprocessed/splits.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def make_folds(
    real_stems: list[str],
    synth_stems: list[str],
    n_folds: int,
    seed: int,
) -> list[dict]:
    """
    Produce n_folds stratified split dicts.

    For 12 images and 5 folds:
      - 2 folds of size 3 (indices 0,1)
      - 3 folds of size 2 (indices 2,3,4)
    No image appears in more than one val set.
    """
    rng = random.Random(seed)
    shuffled = real_stems[:]
    rng.shuffle(shuffled)

    # Distribute into n_folds buckets as evenly as possible
    n = len(shuffled)
    fold_sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]

    folds: list[list[str]] = []
    start = 0
    for size in fold_sizes:
        folds.append(shuffled[start: start + size])
        start += size

    result = []
    for fold_idx in range(n_folds):
        val_real = sorted(folds[fold_idx])
        train_real = sorted(s for i, bucket in enumerate(folds) for s in bucket if i != fold_idx)

        # Build experiment configs
        scale_configs = {}
        for count in [40, 80, 120, len(synth_stems)]:
            if count > len(synth_stems):
                continue
            scale_configs[str(count)] = {
                "train": synth_stems[:count],
                "val": val_real,
            }

        experiments = {
            "E1_oracle": {
                "train": train_real,
                "val": val_real,
                "description": "Train on real data only; upper bound"
            },
            "E2_synth_only": {
                "train": synth_stems,
                "val": val_real,
                "description": "Train on all synthetic; core claim"
            },
            "E3_mixed": {
                "train": train_real + synth_stems,
                "val": val_real,
                "description": "Train on real + synthetic; practical best"
            },
            "E4_scale": scale_configs,
        }

        result.append({
            "fold": fold_idx,
            "real_val": val_real,
            "real_train": train_real,
            "synth_train": synth_stems,
            "experiments": experiments,
        })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5-fold CV splits for P&ID ML experiments."
    )
    parser.add_argument(
        "--real-dir", type=Path, required=True,
        help="Directory containing real GraphML/PNG pairs (e.g. data/PID2Graph OPEN100)"
    )
    parser.add_argument(
        "--synth-dir", type=Path, default=None,
        help="Directory containing synthetic GraphML/PNG pairs"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output splits JSON file path"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fold shuffling (default: 42)"
    )
    parser.add_argument(
        "--graphml-only", action="store_true",
        help="Discover stems from .graphml files only (ignore .png)"
    )
    args = parser.parse_args()

    # Discover real image stems
    real_stems = sorted(
        p.stem for p in args.real_dir.glob("*.graphml")
    )
    if not real_stems:
        log.error("No .graphml files found in %s", args.real_dir)
        return
    log.info("Real images found: %d  (%s)", len(real_stems), ", ".join(real_stems))

    if len(real_stems) < args.n_folds:
        log.error(
            "Cannot create %d folds from only %d real images.",
            args.n_folds, len(real_stems)
        )
        return

    # Discover synthetic stems
    synth_stems: list[str] = []
    if args.synth_dir:
        synth_stems = sorted(
            p.stem for p in args.synth_dir.glob("*.graphml")
        )
        log.info("Synthetic images found: %d", len(synth_stems))
    else:
        log.info("No --synth-dir provided; E2/E3/E4 experiments will have empty train sets")

    # Build folds
    folds = make_folds(
        real_stems=real_stems,
        synth_stems=synth_stems,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    output = {
        "description": (
            "5-fold CV splits for P&ID synthetic-to-real transfer experiments. "
            "Real images are used for evaluation only. "
            "Synthetic images are used for training only. "
            "See docs/research_plan.md for full experiment definitions."
        ),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "real_dir": str(args.real_dir),
        "synth_dir": str(args.synth_dir) if args.synth_dir else None,
        "real_images": real_stems,
        "synthetic_images": synth_stems,
        "folds": folds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info("Saved splits → %s", args.output)

    # Print summary
    log.info("")
    log.info("Fold summary:")
    for fold in folds:
        log.info(
            "  Fold %d: val=%s  train_real=%d  train_synth=%d",
            fold["fold"],
            fold["real_val"],
            len(fold["real_train"]),
            len(fold["synth_train"]),
        )


if __name__ == "__main__":
    main()
