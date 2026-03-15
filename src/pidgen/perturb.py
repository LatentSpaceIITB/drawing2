from __future__ import annotations

import random
from typing import cast

from .complexity import get_profile
from .motifs import (
    apply_bypass_links,
    apply_edge_chain_enrichment,
    apply_label_remix,
    apply_motif_enrichment,
    apply_terminal_expansion,
)
from .scene import BoundingBox, Edge, Node, Scene
from .template import (
    BOTTOM_MARGIN,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CONTENT_BOX,
    LEFT_MARGIN,
    RIGHT_PANEL_WIDTH,
    TOP_MARGIN,
)


def _build_axis_mapping(
    positions: list[int],
    lower: int,
    upper: int,
    rng: random.Random,
    jitter: int,
) -> dict[int, int]:
    if not positions:
        return {}

    mapped: dict[int, float] = {}
    first = positions[0]
    mapped[first] = first + rng.randint(-jitter, jitter)
    for previous, current in zip(positions, positions[1:]):
        gap = current - previous
        scale = 0.84 + (rng.random() * 0.36)
        new_gap = max(12, int(round((gap * scale) / 4.0)) * 4)
        mapped[current] = mapped[previous] + new_gap

    min_value = min(mapped.values())
    max_value = max(mapped.values())
    span = max(max_value - min_value, 1.0)
    target_span = max(upper - lower, 1)
    scale = min(1.0, target_span / span)
    offset = lower + ((target_span - (span * scale)) / 2.0) - (min_value * scale)

    normalized: dict[int, int] = {}
    for original in positions:
        value = int(round((mapped[original] * scale + offset) / 4.0) * 4)
        normalized[original] = min(upper, max(lower, value))
    return normalized


def _reset_backgrounds(scene: Scene) -> None:
    background_nodes = sorted(scene.nodes_by_label("background"), key=lambda node: node.id)
    boxes = [
        BoundingBox(0, 0, LEFT_MARGIN, CANVAS_HEIGHT),
        BoundingBox(0, 0, CANVAS_WIDTH, TOP_MARGIN),
        BoundingBox(CANVAS_WIDTH - RIGHT_PANEL_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT),
        BoundingBox(0, CANVAS_HEIGHT - BOTTOM_MARGIN, CANVAS_WIDTH, CANVAS_HEIGHT),
    ]
    for node, bbox in zip(background_nodes, boxes):
        node.bbox = bbox


def _label_scale(label: str, rng: random.Random, scale_min: float, scale_max: float) -> tuple[float, float]:
    if label in {"connector", "crossing"}:
        return (1.0, 1.0)
    if label == "tank":
        scale = max(0.78, scale_min - 0.04) + (rng.random() * max(0.08, scale_max - scale_min))
        return (scale, min(scale_max + 0.08, scale + 0.12))
    if label == "pump":
        scale = scale_min + (rng.random() * (scale_max - scale_min))
        return (scale, scale)
    if label == "inlet/outlet":
        return (0.9 + (rng.random() * 0.2), 0.9 + (rng.random() * 0.12))
    if label == "instrumentation":
        scale = 0.92 + (rng.random() * max(0.1, scale_max - 0.9))
        return (scale, scale)
    if label in {"valve", "arrow"}:
        scale = scale_min + (rng.random() * (scale_max - scale_min))
        return (scale, scale)
    return (
        scale_min + (rng.random() * (scale_max - scale_min)),
        scale_min + (rng.random() * (scale_max - scale_min)),
    )


def _rescale_bbox(node: Node, rng: random.Random, scale_min: float, scale_max: float) -> None:
    sx, sy = _label_scale(node.label, rng, scale_min, scale_max)
    node.bbox = node.bbox.scale_about_center(sx, sy)


def _edge_length(scene: Scene, edge: Edge) -> float:
    source = scene.nodes[edge.source]
    target = scene.nodes[edge.target]
    return abs(source.bbox.cx - target.bbox.cx) + abs(source.bbox.cy - target.bbox.cy)


def _route_midpoint(scene: Scene, edge: Edge) -> tuple[float, float]:
    source = scene.nodes[edge.source]
    target = scene.nodes[edge.target]
    sx, sy = source.bbox.cx, source.bbox.cy
    tx, ty = target.bbox.cx, target.bbox.cy
    if abs(sx - tx) < 1 or abs(sy - ty) < 1:
        return ((sx + tx) / 2.0, (sy + ty) / 2.0)
    if abs(sx - tx) >= abs(sy - ty):
        return ((sx + tx) / 2.0, sy)
    return (sx, (sy + ty) / 2.0)


def _inline_box(label: str, cx: float, cy: float, rng: random.Random) -> BoundingBox:
    if label == "valve":
        size = rng.randint(42, 72)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "arrow":
        size = rng.randint(36, 56)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "instrumentation":
        size = rng.randint(70, 118)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "pump":
        width = rng.randint(80, 130)
        height = rng.randint(55, 95)
        return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    width = rng.randint(34, 92)
    height = rng.randint(24, 72)
    return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _overlaps_existing(scene: Scene, bbox: BoundingBox, tolerance: float = 6.0) -> bool:
    for node in scene.non_background_nodes():
        if not (
            bbox.xmax < node.bbox.xmin - tolerance
            or bbox.xmin > node.bbox.xmax + tolerance
            or bbox.ymax < node.bbox.ymin - tolerance
            or bbox.ymin > node.bbox.ymax + tolerance
        ):
            return True
    return False


def _next_node_id(scene: Scene, prefix: str) -> str:
    max_index = 0
    for node_id in scene.nodes:
        if node_id.startswith(prefix):
            suffix = node_id[len(prefix) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return f"{prefix}{max_index + 1}"


def _insert_inline_nodes(scene: Scene, rng: random.Random, insertion_range: tuple[int, int]) -> None:
    target_insertions = rng.randint(*insertion_range)
    candidates = [
        edge
        for edge in scene.edges
        if edge.source in scene.nodes
        and edge.target in scene.nodes
        and scene.nodes[edge.source].label != "background"
        and scene.nodes[edge.target].label != "background"
        and _edge_length(scene, edge) >= 220
    ]
    rng.shuffle(candidates)
    insertions = 0
    new_edges: list[Edge] = []
    removed: set[int] = set()

    for edge in candidates:
        if insertions >= target_insertions:
            break
        try:
            edge_index = scene.edges.index(edge)
        except ValueError:
            continue
        cx, cy = _route_midpoint(scene, edge)
        label_pool = ["general", "valve", "arrow", "instrumentation"]
        weights = [0.34, 0.28, 0.12, 0.18]
        if scene.metadata.get("source_dataset") == "open100":
            label_pool.append("pump")
            weights.append(0.08)
            if rng.random() < 0.4:
                label_pool.append("general")
                weights.append(0.1)
        label = rng.choices(label_pool, weights=weights, k=1)[0]
        bbox = _inline_box(label, cx, cy, rng).clamp_inside(CONTENT_BOX, padding=8.0)
        if _overlaps_existing(scene, bbox):
            continue

        node_id = _next_node_id(scene, label.replace("/", "_"))
        scene.nodes[node_id] = Node(id=node_id, label=label, bbox=bbox, metadata={"generated": True})
        new_edges.append(Edge(source=edge.source, target=node_id, style=edge.style))
        new_edges.append(Edge(source=node_id, target=edge.target, style=edge.style))
        removed.add(edge_index)
        insertions += 1

    if removed:
        scene.edges = [edge for index, edge in enumerate(scene.edges) if index not in removed]
        scene.edges.extend(new_edges)


def _prune_leaf_nodes(scene: Scene, rng: random.Random, removal_range: tuple[int, int]) -> None:
    removals = rng.randint(*removal_range)
    if removals <= 0:
        return
    removable_labels = {"instrumentation", "arrow", "general", "valve", "pump", "tank", "inlet/outlet"}
    for _ in range(removals):
        adjacency = scene.adjacency()
        candidates = [
            node
            for node in scene.non_background_nodes()
            if node.label in removable_labels and len(adjacency.get(node.id, [])) <= 1
            and not node.metadata.get("generated_keep")
        ]
        if not candidates:
            break
        node = rng.choice(candidates)
        scene.nodes.pop(node.id, None)
        scene.edges = [edge for edge in scene.edges if edge.source != node.id and edge.target != node.id]


def _retune_edge_styles(scene: Scene, rng: random.Random, target_rate: float) -> None:
    for edge in scene.edges:
        edge.style = "solid"
    candidates = [edge for edge in scene.edges if scene.nodes[edge.source].label != "background" and scene.nodes[edge.target].label != "background"]
    rng.shuffle(candidates)
    desired = int(round(len(candidates) * target_rate))
    for edge in candidates[:desired]:
        edge.style = "non-solid"


def _normalize_open100_scene(scene: Scene, rng: random.Random, jitter: int) -> None:
    content_value = scene.metadata.get("source_content_bbox", (0, 0, scene.width, scene.height))
    content_tuple: tuple[float, float, float, float]
    if isinstance(content_value, (tuple, list)) and len(content_value) == 4:
        content_tuple = cast(tuple[float, float, float, float], tuple(float(value) for value in content_value))
    else:
        content_tuple = (0.0, 0.0, float(scene.width), float(scene.height))
    source_content = BoundingBox(*content_tuple)
    target = CONTENT_BOX
    base_scale = min(target.width / max(source_content.width, 1.0), target.height / max(source_content.height, 1.0))
    scale = base_scale * (0.9 + (rng.random() * 0.12))
    target_width = source_content.width * scale
    target_height = source_content.height * scale
    offset_x = target.xmin + ((target.width - target_width) / 2.0) + rng.randint(-jitter * 2, jitter * 2)
    offset_y = target.ymin + ((target.height - target_height) / 2.0) + rng.randint(-jitter * 2, jitter * 2)

    for node in scene.non_background_nodes():
        bbox = BoundingBox(
            offset_x + ((node.bbox.xmin - source_content.xmin) * scale),
            offset_y + ((node.bbox.ymin - source_content.ymin) * scale),
            offset_x + ((node.bbox.xmax - source_content.xmin) * scale),
            offset_y + ((node.bbox.ymax - source_content.ymin) * scale),
        )
        node.bbox = bbox.clamp_inside(CONTENT_BOX, padding=6.0)

    for node in scene.nodes_by_label("inlet/outlet"):
        side = str(node.metadata.get("anchor_side", "left"))
        cx = node.bbox.cx
        cy = node.bbox.cy
        if side == "left":
            node.bbox = node.bbox.centered_at(CONTENT_BOX.xmin + (node.bbox.width / 2.0) + 8.0, cy)
        elif side == "right":
            node.bbox = node.bbox.centered_at(CONTENT_BOX.xmax - (node.bbox.width / 2.0) - 8.0, cy)
        elif side == "top":
            node.bbox = node.bbox.centered_at(cx, CONTENT_BOX.ymin + (node.bbox.height / 2.0) + 8.0)
        else:
            node.bbox = node.bbox.centered_at(cx, CONTENT_BOX.ymax - (node.bbox.height / 2.0) - 8.0)
        node.bbox = node.bbox.clamp_inside(CONTENT_BOX, padding=6.0)


def _rank_remap_scene(scene: Scene, rng: random.Random, jitter: int) -> None:
    content_nodes = scene.non_background_nodes()
    unique_x = sorted({int(round(node.bbox.cx)) for node in content_nodes})
    unique_y = sorted({int(round(node.bbox.cy)) for node in content_nodes})
    x_map = _build_axis_mapping(unique_x, int(CONTENT_BOX.xmin + 36), int(CONTENT_BOX.xmax - 36), rng, jitter)
    y_map = _build_axis_mapping(unique_y, int(CONTENT_BOX.ymin + 36), int(CONTENT_BOX.ymax - 36), rng, jitter)
    for node in content_nodes:
        current_x = int(round(node.bbox.cx))
        current_y = int(round(node.bbox.cy))
        node.bbox = node.bbox.centered_at(x_map.get(current_x, current_x), y_map.get(current_y, current_y)).clamp_inside(CONTENT_BOX, padding=4.0)


def perturb_scene(scene: Scene, seed: int = 1401, profile_name: str = "medium") -> Scene:
    rng = random.Random(seed)
    profile = get_profile(profile_name)
    new_scene = scene.clone()
    new_scene.width = CANVAS_WIDTH
    new_scene.height = CANVAS_HEIGHT
    new_scene.metadata["generation_seed"] = seed
    new_scene.metadata["generation_profile"] = profile.name

    if new_scene.metadata.get("source_dataset") == "open100":
        _normalize_open100_scene(new_scene, rng, profile.coord_jitter)
    else:
        _rank_remap_scene(new_scene, rng, profile.coord_jitter)

    for node in new_scene.non_background_nodes():
        if node.label == "inlet/outlet":
            continue
        jitter_x = rng.randint(-profile.coord_jitter, profile.coord_jitter)
        jitter_y = rng.randint(-profile.coord_jitter, profile.coord_jitter)
        node.bbox = node.bbox.translate(jitter_x, jitter_y)
        _rescale_bbox(node, rng, profile.scale_min, profile.scale_max)
        node.bbox = node.bbox.clamp_inside(CONTENT_BOX, padding=4.0)

    _prune_leaf_nodes(new_scene, rng, profile.leaf_removals)
    _insert_inline_nodes(new_scene, rng, profile.inline_insertions)
    apply_motif_enrichment(new_scene, rng, CONTENT_BOX, rng.randint(*profile.motif_budget))
    apply_edge_chain_enrichment(new_scene, rng, CONTENT_BOX, rng.randint(*profile.edge_chain_budget))
    apply_terminal_expansion(new_scene, rng, CONTENT_BOX, rng.randint(*profile.terminal_budget))
    apply_bypass_links(new_scene, rng, CONTENT_BOX, rng.randint(*profile.bypass_budget))
    apply_label_remix(new_scene, rng, rng.randint(*profile.relabel_budget))
    _retune_edge_styles(new_scene, rng, profile.non_solid_rate)
    _reset_backgrounds(new_scene)
    return new_scene
