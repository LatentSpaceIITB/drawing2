from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pidgen import (
    DiversityTracker,
    complexity_bucket,
    complexity_score,
    derive_thresholds,
    generate_scene,
    graph_fingerprint,
    parse_graphml,
    write_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a balanced large batch from OPEN100 structural seeds.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "open100_batch.json"),
        help="OPEN100 batch configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the batch output directory.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Optional override for accepted sample count.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Optional override for total generation attempts.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print a progress heartbeat every N attempts.",
    )
    return parser


def _deterministic_seed(master_seed: int, attempt: int, seed_path: Path, bucket: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{attempt}:{seed_path.name}:{bucket}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pick_generation_profile(target_bucket: str, profile_variants: dict[str, dict[str, int]], master_seed: int, attempt: int) -> str:
    weighted_profiles = profile_variants.get(target_bucket, {target_bucket: 100})
    total = max(sum(int(weight) for weight in weighted_profiles.values()), 1)
    token = int(hashlib.sha256(f"profile:{master_seed}:{attempt}:{target_bucket}".encode("utf-8")).hexdigest()[:8], 16)
    cursor = token % total
    for profile_name, weight in weighted_profiles.items():
        weight_value = int(weight)
        if cursor < weight_value:
            return str(profile_name)
        cursor -= weight_value
    return target_bucket


def _pick_bucket(remaining: dict[str, int], attempt: int) -> str:
    ordered = sorted(remaining.items(), key=lambda item: (-item[1], item[0]))
    positives = [bucket for bucket, count in ordered if count > 0]
    if not positives:
        return ordered[0][0] if ordered else "simple"
    return positives[attempt % len(positives)]


def _ordered_buckets(target_bucket: str, available_buckets: list[str]) -> list[str]:
    fallbacks = {
        "simple": ["simple", "medium", "dense"],
        "medium": ["medium", "simple", "dense"],
        "dense": ["dense", "medium", "simple"],
    }
    order = fallbacks.get(target_bucket, [target_bucket, "simple", "dense", "medium"])
    ordered = [bucket for bucket in order if bucket in available_buckets]
    for bucket in available_buckets:
        if bucket not in ordered:
            ordered.append(bucket)
    return ordered


def _pick_seed_for_bucket(seed_infos: list[dict[str, object]], target_bucket: str, attempt: int, available_buckets: list[str]) -> Path:
    ordered: list[dict[str, object]] = []
    for bucket in _ordered_buckets(target_bucket, available_buckets):
        ordered.extend(sorted((info for info in seed_infos if info["bucket"] == bucket), key=lambda info: str(info["path"])))
    if not ordered:
        ordered = sorted(seed_infos, key=lambda info: str(info["path"]))
    return Path(str(ordered[attempt % len(ordered)]["path"]))


def _delete_manifest_outputs(manifest: dict[str, object]) -> None:
    outputs = manifest.get("outputs", {})
    if isinstance(outputs, dict):
        for path_text in outputs.values():
            path = Path(str(path_text))
            if path.exists():
                path.unlink()
    manifest_path = Path(str(manifest.get("manifest", "")))
    if manifest_path.exists():
        manifest_path.unlink()


def _quota_counts(target_count: int, weights: dict[str, int]) -> dict[str, int]:
    total_weight = max(sum(weights.values()), 1)
    counts = {
        bucket: int(target_count * weight / total_weight)
        for bucket, weight in weights.items()
        if weight > 0
    }
    missing = target_count - sum(counts.values())
    ordered_buckets = sorted(counts, key=lambda bucket: (-counts[bucket], bucket))
    for bucket in ordered_buckets:
        if missing <= 0:
            break
        counts[bucket] = counts.get(bucket, 0) + 1
        missing -= 1
    return counts


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / str(config.get("output_dir", "outputs/open100_structural"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = ROOT / str(config.get("seed_dir", "data/PID2Graph OPEN100"))
    seed_pattern = str(config.get("seed_pattern", "*.graphml"))
    seed_paths = sorted(seed_dir.glob(seed_pattern))
    if not seed_paths:
        raise SystemExit(f"no seed GraphML files found in {seed_dir}")

    target_count = int(args.target_count or config.get("target_count", 1000))
    max_attempts = int(args.max_attempts or config.get("max_attempts", target_count * 20))
    progress_every = max(1, int(args.progress_every))
    master_seed = int(config.get("master_seed", 20260315))
    quotas = _quota_counts(target_count, dict(config.get("complexity_weights", {"simple": 25, "medium": 50, "dense": 25})))
    profile_variants = {
        str(bucket): {str(profile_name): int(weight) for profile_name, weight in variants.items()}
        for bucket, variants in dict(config.get("profile_variants", {})).items()
    }
    allowed_buckets = sorted(quotas)
    diversity_cfg = dict(config.get("diversity", {}))
    epoch_size = int(diversity_cfg.get("epoch_size", 500))
    epoch_prime = int(diversity_cfg.get("epoch_prime", 97))

    seed_infos: list[dict[str, object]] = []
    seed_scores: list[float] = []
    for seed_path in seed_paths:
        scene = parse_graphml(seed_path)
        score = complexity_score(scene)
        seed_scores.append(score)
        seed_infos.append({"path": seed_path, "score": score})
    thresholds = derive_thresholds(seed_scores)
    for seed_info in seed_infos:
        seed_scene = parse_graphml(Path(str(seed_info["path"])))
        seed_info["bucket"] = complexity_bucket(seed_scene, thresholds)

    accepted: list[dict[str, object]] = []
    rejections = {"bucket_mismatch": 0, "graph_duplicate": 0, "image_duplicate": 0}
    accepted_profiles: Counter[str] = Counter()
    tracker = DiversityTracker(
        min_hist_distance=int(diversity_cfg.get("min_hist_distance", 12)),
        max_same_seed_image_distance=int(diversity_cfg.get("max_same_seed_image_distance", 6)),
        max_global_image_distance=int(diversity_cfg.get("max_global_image_distance", 3)),
    )
    remaining = dict(quotas)

    attempt = 0
    while len(accepted) < target_count and attempt < max_attempts:
        if attempt > 0 and attempt % progress_every == 0:
            print(
                f"progress attempts={attempt}/{max_attempts} accepted={len(accepted)}/{target_count} "
                f"bucket_mismatch={rejections['bucket_mismatch']} graph_duplicate={rejections['graph_duplicate']} "
                f"image_duplicate={rejections['image_duplicate']}"
            )
        target_bucket = _pick_bucket(remaining, attempt)
        seed_path = _pick_seed_for_bucket(seed_infos, target_bucket, attempt, ["simple", "medium", "dense"])
        requested_profile = _pick_generation_profile(target_bucket, profile_variants, master_seed, attempt)
        epoch = attempt // epoch_size
        effective_master_seed = master_seed + (epoch * epoch_prime)
        generation_seed = _deterministic_seed(effective_master_seed, attempt, seed_path, target_bucket)
        scene = generate_scene(seed_graphml=seed_path, seed=generation_seed, profile_name=requested_profile)
        actual_bucket = complexity_bucket(scene, thresholds)
        accepted_bucket = target_bucket
        if actual_bucket != target_bucket:
            if actual_bucket in allowed_buckets and remaining.get(actual_bucket, 0) > 0:
                accepted_bucket = actual_bucket
            else:
                rejections["bucket_mismatch"] += 1
                attempt += 1
                continue
        if remaining.get(accepted_bucket, 0) <= 0:
            rejections["bucket_mismatch"] += 1
            attempt += 1
            continue

        fingerprint = scene.metadata.get("graph_fingerprint")
        if fingerprint is None:
            fingerprint = graph_fingerprint(scene)
            scene.metadata["graph_fingerprint"] = fingerprint
        if not tracker.accept_graph(fingerprint):
            rejections["graph_duplicate"] += 1
            attempt += 1
            continue

        stem = f"open100_{accepted_bucket}_{len(accepted):05d}"
        manifest = write_outputs(
            scene=scene,
            seed_graphml=seed_path,
            output_dir=output_dir,
            stem=stem,
            seed=generation_seed,
            profile_name=requested_profile,
        )
        manifest["accepted_bucket"] = accepted_bucket
        manifest["requested_profile"] = requested_profile
        manifest["requested_bucket"] = target_bucket
        scene_summary = manifest.get("scene_summary")
        if isinstance(scene_summary, dict):
            scene_summary["accepted_bucket"] = accepted_bucket
            scene_summary["requested_profile"] = requested_profile
            scene_summary["requested_bucket"] = target_bucket
        Path(str(manifest["manifest"])).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        image_hashes = manifest.get("image_hashes", {})
        ahash = str(image_hashes.get("average_hash", "0"))
        dhash = str(image_hashes.get("difference_hash", "0"))
        if not tracker.accept_image(str(scene.metadata.get("seed_name", seed_path.stem)), ahash, dhash):
            rejections["image_duplicate"] += 1
            _delete_manifest_outputs(manifest)
            attempt += 1
            continue

        accepted.append(manifest)
        accepted_profiles[requested_profile] += 1
        remaining[accepted_bucket] = max(0, remaining.get(accepted_bucket, 0) - 1)
        print(f"accepted {stem} seed={seed_path.stem} bucket={accepted_bucket} profile={requested_profile} count={len(accepted)}/{target_count}")
        attempt += 1

    batch_manifest = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "seed_dir": str(seed_dir),
        "target_count": target_count,
        "accepted_count": len(accepted),
        "attempts": attempt,
        "max_attempts": max_attempts,
        "master_seed": master_seed,
        "epoch_size": epoch_size,
        "epoch_prime": epoch_prime,
        "diversity_config": diversity_cfg,
        "complexity_thresholds": {"lower": thresholds[0], "upper": thresholds[1]},
        "quota_plan": quotas,
        "quota_remaining": remaining,
        "profile_variants": profile_variants,
        "accepted_profile_counts": dict(sorted(accepted_profiles.items())),
        "rejections": rejections,
        "jobs": accepted,
    }
    batch_manifest_path = output_dir / "batch_manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    print(f"batch manifest: {batch_manifest_path}")


if __name__ == "__main__":
    main()
