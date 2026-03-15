from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

from .annotate import annotate_scene
from .diversity import average_hash, difference_hash
from .export_graphml import export_graphml
from .perturb import perturb_scene
from .renderers import render_png, render_svg
from .scene import Scene
from .seed_parser import parse_graphml
from .stats import scene_summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_scene(seed_graphml: str | Path, seed: int, profile_name: str = "medium") -> Scene:
    scene = parse_graphml(seed_graphml)
    scene = perturb_scene(scene, seed=seed, profile_name=profile_name)
    annotate_scene(scene, seed=seed, profile_name=profile_name)
    return scene


def write_outputs(
    scene: Scene,
    seed_graphml: str | Path,
    output_dir: str | Path,
    stem: str,
    seed: int,
    profile_name: str = "medium",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    graphml_path = output_dir / f"{stem}.graphml"
    manifest_path = output_dir / f"{stem}.manifest.json"

    render_png(scene, png_path)
    render_svg(scene, svg_path)
    export_graphml(scene, graphml_path)

    manifest = {
        "stem": stem,
        "seed": seed,
        "profile_name": profile_name,
        "seed_graphml": str(Path(seed_graphml)),
        "outputs": {
            "png": str(png_path),
            "svg": str(svg_path),
            "graphml": str(graphml_path),
        },
        "image_hashes": {
            "average_hash": average_hash(png_path),
            "difference_hash": difference_hash(png_path),
        },
        "sha256": {
            "png": _sha256(png_path),
            "svg": _sha256(svg_path),
            "graphml": _sha256(graphml_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scene_summary": scene_summary(scene),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def generate_outputs(seed_graphml: str | Path, output_dir: str | Path, stem: str, seed: int, profile_name: str = "medium") -> dict[str, object]:
    scene: Scene = generate_scene(seed_graphml=seed_graphml, seed=seed, profile_name=profile_name)
    return write_outputs(
        scene=scene,
        seed_graphml=seed_graphml,
        output_dir=output_dir,
        stem=stem,
        seed=seed,
        profile_name=profile_name,
    )
