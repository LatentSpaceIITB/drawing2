from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pidgen import generate_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one phase-1 synthetic P&ID sample.")
    parser.add_argument(
        "--seed-graphml",
        default=str(ROOT / "data" / "Dataset PID" / "0.graphml"),
        help="Bootstrap GraphML seed file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "phase1"),
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1401,
        help="Random seed controlling perturbation and text generation.",
    )
    parser.add_argument(
        "--stem",
        default="phase1_sample",
        help="Output file stem without extension.",
    )
    parser.add_argument(
        "--profile",
        default="medium",
        choices=["simple", "medium", "dense", "dense_standard", "dense_heavy"],
        help="Complexity profile for generation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seed_graphml = Path(args.seed_graphml)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = generate_outputs(
        seed_graphml=seed_graphml,
        output_dir=output_dir,
        stem=args.stem,
        seed=args.seed,
        profile_name=args.profile,
    )

    print(f"PNG: {manifest['outputs']['png']}")
    print(f"SVG: {manifest['outputs']['svg']}")
    print(f"GraphML: {manifest['outputs']['graphml']}")
    print(f"Manifest: {manifest['manifest']}")


if __name__ == "__main__":
    main()
