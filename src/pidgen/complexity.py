from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import quantiles
from typing import Iterable

from .scene import Scene


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    coord_jitter: int
    scale_min: float
    scale_max: float
    inline_insertions: tuple[int, int]
    leaf_removals: tuple[int, int]
    motif_budget: tuple[int, int]
    edge_chain_budget: tuple[int, int]
    bypass_budget: tuple[int, int]
    terminal_budget: tuple[int, int]
    relabel_budget: tuple[int, int]
    non_solid_rate: float
    edge_label_rate: float
    note_count: int
    title_density: str


PROFILE_PRESETS = {
    "simple": GenerationProfile(
        name="simple",
        coord_jitter=18,
        scale_min=0.86,
        scale_max=1.05,
        inline_insertions=(1, 3),
        leaf_removals=(8, 20),
        motif_budget=(0, 1),
        edge_chain_budget=(0, 1),
        bypass_budget=(0, 0),
        terminal_budget=(0, 1),
        relabel_budget=(0, 1),
        non_solid_rate=0.03,
        edge_label_rate=0.14,
        note_count=10,
        title_density="light",
    ),
    "medium": GenerationProfile(
        name="medium",
        coord_jitter=30,
        scale_min=0.84,
        scale_max=1.14,
        inline_insertions=(4, 8),
        leaf_removals=(0, 6),
        motif_budget=(1, 3),
        edge_chain_budget=(1, 4),
        bypass_budget=(0, 2),
        terminal_budget=(1, 2),
        relabel_budget=(1, 3),
        non_solid_rate=0.05,
        edge_label_rate=0.22,
        note_count=14,
        title_density="normal",
    ),
    "dense": GenerationProfile(
        name="dense",
        coord_jitter=42,
        scale_min=0.82,
        scale_max=1.22,
        inline_insertions=(8, 16),
        leaf_removals=(0, 2),
        motif_budget=(3, 6),
        edge_chain_budget=(4, 8),
        bypass_budget=(2, 5),
        terminal_budget=(2, 4),
        relabel_budget=(2, 4),
        non_solid_rate=0.08,
        edge_label_rate=0.31,
        note_count=16,
        title_density="dense",
    ),
    "dense_standard": GenerationProfile(
        name="dense_standard",
        coord_jitter=38,
        scale_min=0.83,
        scale_max=1.18,
        inline_insertions=(7, 13),
        leaf_removals=(0, 1),
        motif_budget=(2, 5),
        edge_chain_budget=(3, 7),
        bypass_budget=(1, 4),
        terminal_budget=(2, 3),
        relabel_budget=(1, 3),
        non_solid_rate=0.07,
        edge_label_rate=0.27,
        note_count=15,
        title_density="dense",
    ),
    "dense_heavy": GenerationProfile(
        name="dense_heavy",
        coord_jitter=50,
        scale_min=0.82,
        scale_max=1.26,
        inline_insertions=(12, 22),
        leaf_removals=(0, 0),
        motif_budget=(5, 9),
        edge_chain_budget=(8, 16),
        bypass_budget=(4, 10),
        terminal_budget=(3, 6),
        relabel_budget=(3, 7),
        non_solid_rate=0.1,
        edge_label_rate=0.36,
        note_count=16,
        title_density="dense",
    ),
    "dense_wide": GenerationProfile(
        name="dense_wide",
        coord_jitter=65,
        scale_min=0.78,
        scale_max=1.30,
        inline_insertions=(6, 12),
        leaf_removals=(0, 1),
        motif_budget=(3, 6),
        edge_chain_budget=(3, 7),
        bypass_budget=(6, 14),
        terminal_budget=(2, 4),
        relabel_budget=(4, 8),
        non_solid_rate=0.09,
        edge_label_rate=0.28,
        note_count=15,
        title_density="dense",
    ),
    "dense_compact": GenerationProfile(
        name="dense_compact",
        coord_jitter=28,
        scale_min=0.86,
        scale_max=1.14,
        inline_insertions=(18, 30),
        leaf_removals=(0, 0),
        motif_budget=(4, 7),
        edge_chain_budget=(12, 22),
        bypass_budget=(2, 5),
        terminal_budget=(4, 8),
        relabel_budget=(5, 10),
        non_solid_rate=0.12,
        edge_label_rate=0.40,
        note_count=16,
        title_density="dense",
    ),
}


def get_profile(name: str) -> GenerationProfile:
    return PROFILE_PRESETS.get(name, PROFILE_PRESETS["medium"])


def profile_dict(name: str) -> dict[str, object]:
    return asdict(get_profile(name))


def complexity_score(scene: Scene) -> float:
    node_count = len(scene.nodes)
    edge_count = len(scene.edges)
    label_diversity = len({node.label for node in scene.non_background_nodes()})
    dashed_count = sum(1 for edge in scene.edges if edge.style != "solid")
    labelled_edges = sum(1 for edge in scene.edges if edge.metadata.get("text"))
    return node_count + (0.65 * edge_count) + (label_diversity * 18.0) + (dashed_count * 3.0) + (labelled_edges * 0.3)


def derive_thresholds(scores: Iterable[float]) -> tuple[float, float]:
    values = sorted(float(score) for score in scores)
    if len(values) < 3:
        return (420.0, 620.0)
    lower, upper = quantiles(values, n=3, method="inclusive")
    return (lower, upper)


def complexity_bucket(scene: Scene, thresholds: tuple[float, float] | None = None) -> str:
    lower, upper = thresholds or (420.0, 620.0)
    score = complexity_score(scene)
    if score <= lower:
        return "simple"
    if score >= upper:
        return "dense"
    return "medium"
