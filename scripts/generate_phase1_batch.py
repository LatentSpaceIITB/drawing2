from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pidgen import generate_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a reproducible batch of phase-1 synthetic P&ID samples.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "phase1_batch.json"),
        help="Batch configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the batch output directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_output_dir = Path(args.output_dir) if args.output_dir else ROOT / config.get("output_dir", "outputs/phase1_batch")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, object]] = []
    for job in config.get("jobs", []):
        seed_graphml = ROOT / str(job["seed_graphml"])
        stem = str(job["stem"])
        seed = int(job["seed"])
        profile_name = str(job.get("profile_name", "medium"))
        manifest = generate_outputs(
            seed_graphml=seed_graphml,
            output_dir=base_output_dir,
            stem=stem,
            seed=seed,
            profile_name=profile_name,
        )
        manifests.append(manifest)
        print(f"generated {stem} from {seed_graphml.name} seed={seed} profile={profile_name}")

    batch_manifest = {
        "config": str(config_path),
        "output_dir": str(base_output_dir),
        "jobs": manifests,
    }
    batch_manifest_path = base_output_dir / "batch_manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    print(f"batch manifest: {batch_manifest_path}")


if __name__ == "__main__":
    main()
