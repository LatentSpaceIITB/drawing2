from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from PIL import Image

from .scene import Scene


def _bits_to_hex(bits: list[int]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    width = max(1, len(bits) // 4)
    return f"{value:0{width}x}"


def average_hash(image_path: str | Path, size: int = 8) -> str:
    image = Image.open(image_path).convert("L").resize((size, size))
    pixels = list(image.tobytes())
    mean = sum(pixels) / max(len(pixels), 1)
    bits = [1 if pixel >= mean else 0 for pixel in pixels]
    return _bits_to_hex(bits)


def difference_hash(image_path: str | Path, size: int = 8) -> str:
    image = Image.open(image_path).convert("L").resize((size + 1, size))
    pixels = list(image.tobytes())
    bits: list[int] = []
    for row in range(size):
        start = row * (size + 1)
        for col in range(size):
            bits.append(1 if pixels[start + col] >= pixels[start + col + 1] else 0)
    return _bits_to_hex(bits)


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def graph_fingerprint(scene: Scene) -> dict[str, object]:
    graph = nx.Graph()
    adjacency = scene.adjacency()
    for node in scene.sorted_nodes():
        degree_bucket = min(len(adjacency.get(node.id, [])), 4)
        graph.add_node(node.id, label=f"{node.label}:{degree_bucket}")
    for index, edge in enumerate(scene.edges):
        graph.add_edge(edge.source, edge.target, style=edge.style, key=str(index))

    wl_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="style")
    label_hist = Counter(node.label for node in scene.non_background_nodes())
    degree_hist = Counter(min(len(adjacency.get(node.id, [])), 5) for node in scene.non_background_nodes())
    return {
        "wl_hash": wl_hash,
        "label_hist": dict(sorted(label_hist.items())),
        "degree_hist": dict(sorted(degree_hist.items())),
        "node_count": len(scene.nodes),
        "edge_count": len(scene.edges),
        "source_dataset": scene.metadata.get("source_dataset", "unknown"),
        "seed_name": scene.metadata.get("seed_name", ""),
    }


def histogram_distance(left: dict[str, int], right: dict[str, int]) -> int:
    keys = set(left) | set(right)
    return sum(abs(int(left.get(key, 0)) - int(right.get(key, 0))) for key in keys)


@dataclass
class DiversityTracker:
    graph_hashes: set[str] = field(default_factory=set)
    fingerprints: list[dict[str, object]] = field(default_factory=list)
    image_hashes: list[tuple[str, str, str]] = field(default_factory=list)
    min_hist_distance: int = 12
    max_same_seed_image_distance: int = 6
    max_global_image_distance: int = 3

    def accept_graph(self, fingerprint: dict[str, object]) -> bool:
        wl_hash = str(fingerprint["wl_hash"])
        if wl_hash in self.graph_hashes:
            return False
        raw_hist = fingerprint.get("label_hist", {})
        current_hist: dict[str, int] = raw_hist if isinstance(raw_hist, dict) else {}
        for other in self.fingerprints:
            if other.get("seed_name") == fingerprint.get("seed_name"):
                other_raw = other.get("label_hist", {})
                other_hist: dict[str, int] = other_raw if isinstance(other_raw, dict) else {}
                distance = histogram_distance(current_hist, other_hist)
                if distance < self.min_hist_distance:
                    return False
        self.graph_hashes.add(wl_hash)
        self.fingerprints.append(fingerprint)
        return True

    def accept_image(self, seed_name: str, ahash: str, dhash: str) -> bool:
        for other_seed, other_ahash, other_dhash in self.image_hashes:
            a_dist = hamming_distance(ahash, other_ahash)
            d_dist = hamming_distance(dhash, other_dhash)
            if other_seed == seed_name:
                if a_dist <= self.max_same_seed_image_distance and d_dist <= self.max_same_seed_image_distance:
                    return False
            elif a_dist <= self.max_global_image_distance and d_dist <= self.max_global_image_distance:
                return False
        self.image_hashes.append((seed_name, ahash, dhash))
        return True
