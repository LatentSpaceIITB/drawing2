from __future__ import annotations

from collections import Counter

from .complexity import complexity_bucket, complexity_score
from .diversity import graph_fingerprint
from .scene import Scene


def label_counts(scene: Scene) -> dict[str, int]:
    counter = Counter(node.label for node in scene.nodes.values())
    return dict(sorted(counter.items()))


def edge_style_counts(scene: Scene) -> dict[str, int]:
    counter = Counter(edge.style for edge in scene.edges)
    return dict(sorted(counter.items()))


def scene_summary(scene: Scene) -> dict[str, object]:
    fingerprint = graph_fingerprint(scene)
    return {
        "width": scene.width,
        "height": scene.height,
        "node_count": len(scene.nodes),
        "edge_count": len(scene.edges),
        "label_counts": label_counts(scene),
        "edge_style_counts": edge_style_counts(scene),
        "source_dataset": scene.metadata.get("source_dataset", "unknown"),
        "seed_name": scene.metadata.get("seed_name", ""),
        "generation_seed": scene.metadata.get("generation_seed", None),
        "generation_profile": scene.metadata.get("generation_profile", "medium"),
        "complexity_score": round(complexity_score(scene), 2),
        "complexity_bucket": complexity_bucket(scene),
        "graph_fingerprint": fingerprint,
    }
