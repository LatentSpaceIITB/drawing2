from __future__ import annotations

import random
from math import dist

from .scene import BoundingBox, Edge, Node, Scene


def _next_node_id(scene: Scene, prefix: str) -> str:
    max_index = 0
    for node_id in scene.nodes:
        if node_id.startswith(prefix):
            suffix = node_id[len(prefix) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return f"{prefix}{max_index + 1}"


def _bbox_for_label(label: str, cx: float, cy: float, rng: random.Random) -> BoundingBox:
    if label in {"connector", "crossing"}:
        return BoundingBox(cx - 4, cy - 4, cx + 4, cy + 4)
    if label == "instrumentation":
        size = rng.randint(72, 126)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "valve":
        size = rng.randint(46, 78)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "arrow":
        size = rng.randint(34, 58)
        return BoundingBox(cx - size / 2.0, cy - size / 2.0, cx + size / 2.0, cy + size / 2.0)
    if label == "tank":
        width = rng.randint(110, 190)
        height = rng.randint(140, 260)
        return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    if label == "pump":
        width = rng.randint(90, 150)
        height = rng.randint(60, 120)
        return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    if label == "inlet/outlet":
        width = rng.randint(80, 140)
        height = rng.randint(24, 68)
        return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    width = rng.randint(44, 110)
    height = rng.randint(24, 84)
    return BoundingBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _overlaps(scene: Scene, bbox: BoundingBox, tolerance: float = 10.0, ignore_ids: set[str] | None = None) -> bool:
    ignore_ids = ignore_ids or set()
    for node in scene.non_background_nodes():
        if node.id in ignore_ids:
            continue
        if not (
            bbox.xmax < node.bbox.xmin - tolerance
            or bbox.xmin > node.bbox.xmax + tolerance
            or bbox.ymax < node.bbox.ymin - tolerance
            or bbox.ymin > node.bbox.ymax + tolerance
        ):
            return True
    return False


def _direction_vector(direction: str) -> tuple[int, int]:
    mapping = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
    }
    return mapping[direction]


def _candidate_directions(host: Node, bounds: BoundingBox) -> list[str]:
    spaces = {
        "left": host.bbox.cx - bounds.xmin,
        "right": bounds.xmax - host.bbox.cx,
        "up": host.bbox.cy - bounds.ymin,
        "down": bounds.ymax - host.bbox.cy,
    }
    return [direction for direction, _ in sorted(spaces.items(), key=lambda item: item[1], reverse=True)]


def _make_node(scene: Scene, label: str, bbox: BoundingBox, metadata: dict[str, object] | None = None) -> Node:
    prefix = label.replace("/", "_")
    node = Node(id=_next_node_id(scene, prefix), label=label, bbox=bbox, metadata=metadata or {})
    scene.add_node(node)
    return node


def _route_points(scene: Scene, edge: Edge) -> list[tuple[float, float]]:
    source = scene.nodes[edge.source]
    target = scene.nodes[edge.target]
    sx, sy = source.bbox.cx, source.bbox.cy
    tx, ty = target.bbox.cx, target.bbox.cy
    if abs(sx - tx) < 1 or abs(sy - ty) < 1:
        return [(sx, sy), (tx, ty)]
    if abs(sx - tx) >= abs(sy - ty):
        return [(sx, sy), (tx, sy), (tx, ty)]
    return [(sx, sy), (sx, ty), (tx, ty)]


def _point_on_polyline(points: list[tuple[float, float]], ratio: float) -> tuple[float, float]:
    if len(points) <= 1:
        return points[0]
    segments = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        length = dist(start, end)
        segments.append((start, end, length))
        total += length
    target = total * ratio
    for start, end, length in segments:
        if target <= length:
            segment_ratio = target / max(length, 1.0)
            return (
                start[0] + ((end[0] - start[0]) * segment_ratio),
                start[1] + ((end[1] - start[1]) * segment_ratio),
            )
        target -= length
    return points[-1]


def _nodes_connected(scene: Scene, left_id: str, right_id: str) -> bool:
    for edge in scene.edges:
        if {edge.source, edge.target} == {left_id, right_id}:
            return True
    return False


def _remove_edge(scene: Scene, edge: Edge) -> None:
    try:
        index = scene.edges.index(edge)
    except ValueError:
        return
    scene.edges.pop(index)


def _add_branch_chain(scene: Scene, host: Node, labels: list[str], direction: str, rng: random.Random, bounds: BoundingBox) -> bool:
    dx, dy = _direction_vector(direction)
    cursor_x = host.bbox.cx + (dx * rng.randint(70, 120))
    cursor_y = host.bbox.cy + (dy * rng.randint(70, 120))
    previous = host
    created: list[str] = []
    for label in labels:
        if dx != 0:
            cursor_x += dx * rng.randint(80, 170)
        else:
            cursor_y += dy * rng.randint(80, 170)
        bbox = _bbox_for_label(label, cursor_x, cursor_y, rng).clamp_inside(bounds, padding=6.0)
        if _overlaps(scene, bbox, ignore_ids={host.id, *created}):
            for node_id in created:
                scene.nodes.pop(node_id, None)
            scene.edges = [edge for edge in scene.edges if edge.source not in created and edge.target not in created]
            return False
        node = _make_node(scene, label, bbox, metadata={"generated": True, "generated_keep": True})
        created.append(node.id)
        scene.add_edge(Edge(source=previous.id, target=node.id, style="solid"))
        previous = node
    return True


def _attach_micro_branch(scene: Scene, host: Node, rng: random.Random, bounds: BoundingBox, orientation: str) -> bool:
    direction_choices = ["up", "down"] if orientation == "horizontal" else ["left", "right"]
    rng.shuffle(direction_choices)
    patterns = [["instrumentation"], ["valve", "instrumentation"], ["general", "arrow"]]
    rng.shuffle(patterns)
    for direction in direction_choices:
        for pattern in patterns:
            if _add_branch_chain(scene, host, pattern, direction, rng, bounds):
                return True
    return False


def apply_motif_enrichment(scene: Scene, rng: random.Random, bounds: BoundingBox, budget: int) -> int:
    if budget <= 0:
        return 0
    patterns = [
        ["valve", "instrumentation"],
        ["general", "instrumentation"],
        ["pump", "valve", "arrow"],
        ["tank", "instrumentation"],
        ["inlet/outlet", "arrow"],
        ["general", "valve", "instrumentation"],
        ["general", "general", "arrow"],
    ]
    hosts = [node for node in scene.non_background_nodes() if node.label in {"connector", "crossing", "general", "valve", "pump", "tank"}]
    rng.shuffle(hosts)
    created = 0
    for host in hosts:
        if created >= budget:
            break
        directions = _candidate_directions(host, bounds)
        rng.shuffle(patterns)
        success = False
        for direction in directions[:4]:
            for pattern in patterns:
                if _add_branch_chain(scene, host, pattern, direction, rng, bounds):
                    created += 1
                    success = True
                    break
            if success:
                break
    return created


def apply_edge_chain_enrichment(scene: Scene, rng: random.Random, bounds: BoundingBox, budget: int) -> int:
    if budget <= 0:
        return 0
    patterns = [
        ["valve"],
        ["general", "valve"],
        ["valve", "instrumentation"],
        ["general", "valve", "arrow"],
        ["general", "general", "valve"],
        ["pump", "valve"],
    ]
    candidates = [
        edge
        for edge in list(scene.edges)
        if edge.source in scene.nodes
        and edge.target in scene.nodes
        and scene.nodes[edge.source].label != "background"
        and scene.nodes[edge.target].label != "background"
    ]
    candidates = [edge for edge in candidates if dist((scene.nodes[edge.source].bbox.cx, scene.nodes[edge.source].bbox.cy), (scene.nodes[edge.target].bbox.cx, scene.nodes[edge.target].bbox.cy)) > 180]
    rng.shuffle(candidates)
    created = 0
    for edge in candidates:
        if created >= budget:
            break
        source = scene.nodes.get(edge.source)
        target = scene.nodes.get(edge.target)
        if source is None or target is None:
            continue
        points = _route_points(scene, edge)
        labels = rng.choice(patterns)
        fractions = [float(index + 1) / float(len(labels) + 1) for index in range(len(labels))]
        new_nodes: list[Node] = []
        ignore_ids = {source.id, target.id}
        valid = True
        for label, ratio in zip(labels, fractions):
            cx, cy = _point_on_polyline(points, ratio)
            bbox = _bbox_for_label(label, cx, cy, rng).clamp_inside(bounds, padding=6.0)
            if _overlaps(scene, bbox, tolerance=8.0, ignore_ids=ignore_ids):
                valid = False
                break
            node = _make_node(scene, label, bbox, metadata={"generated": True, "generated_keep": True})
            new_nodes.append(node)
            ignore_ids.add(node.id)
        if not valid:
            for node in new_nodes:
                scene.nodes.pop(node.id, None)
            continue
        _remove_edge(scene, edge)
        previous = source.id
        for node in new_nodes:
            scene.add_edge(Edge(source=previous, target=node.id, style=edge.style))
            previous = node.id
        scene.add_edge(Edge(source=previous, target=target.id, style=edge.style))
        created += len(new_nodes)
        if new_nodes and rng.random() < 0.65:
            orientation = "horizontal" if abs(source.bbox.cx - target.bbox.cx) >= abs(source.bbox.cy - target.bbox.cy) else "vertical"
            _attach_micro_branch(scene, rng.choice(new_nodes), rng, bounds, orientation)
    return created


def apply_terminal_expansion(scene: Scene, rng: random.Random, bounds: BoundingBox, budget: int) -> int:
    if budget <= 0:
        return 0
    inward = {"left": "right", "right": "left", "top": "down", "bottom": "up"}
    patterns = [["valve"], ["general", "valve"], ["instrumentation", "valve"], ["general", "valve", "instrumentation"]]
    created = 0
    terminals = [node for node in scene.nodes_by_label("inlet/outlet") if len(scene.adjacency().get(node.id, [])) >= 1]
    rng.shuffle(terminals)
    for node in terminals:
        if created >= budget:
            break
        adjacency = scene.adjacency().get(node.id, [])
        if not adjacency:
            continue
        neighbor_id = adjacency[0]
        edge = next((candidate for candidate in scene.edges if {candidate.source, candidate.target} == {node.id, neighbor_id}), None)
        if edge is None:
            continue
        direction = inward.get(str(node.metadata.get("anchor_side", "left")), "right")
        labels = rng.choice(patterns)
        dx, dy = _direction_vector(direction)
        previous = node.id
        cursor_x = node.bbox.cx
        cursor_y = node.bbox.cy
        new_nodes: list[Node] = []
        valid = True
        ignore_ids = {node.id, neighbor_id}
        for label in labels:
            cursor_x += dx * rng.randint(80, 150)
            cursor_y += dy * rng.randint(80, 150)
            bbox = _bbox_for_label(label, cursor_x, cursor_y, rng).clamp_inside(bounds, padding=6.0)
            if _overlaps(scene, bbox, tolerance=8.0, ignore_ids=ignore_ids):
                valid = False
                break
            new_node = _make_node(scene, label, bbox, metadata={"generated": True, "generated_keep": True})
            new_nodes.append(new_node)
            ignore_ids.add(new_node.id)
        if not valid:
            for created_node in new_nodes:
                scene.nodes.pop(created_node.id, None)
            continue
        _remove_edge(scene, edge)
        for new_node in new_nodes:
            scene.add_edge(Edge(source=previous, target=new_node.id, style=edge.style))
            previous = new_node.id
        scene.add_edge(Edge(source=previous, target=neighbor_id, style=edge.style))
        created += len(new_nodes)
    return created


def apply_bypass_links(scene: Scene, rng: random.Random, bounds: BoundingBox, budget: int) -> int:
    if budget <= 0:
        return 0
    labels = ["valve", "general", "connector"]
    nodes = [node for node in scene.non_background_nodes() if node.label in {"connector", "crossing", "general", "valve", "pump", "tank"}]
    rng.shuffle(nodes)
    created = 0
    for left in nodes:
        if created >= budget:
            break
        for right in nodes:
            if created >= budget:
                break
            if left.id == right.id or _nodes_connected(scene, left.id, right.id):
                continue
            same_row = abs(left.bbox.cy - right.bbox.cy) <= 90
            same_col = abs(left.bbox.cx - right.bbox.cx) <= 90
            if not same_row and not same_col:
                continue
            manhattan = abs(left.bbox.cx - right.bbox.cx) + abs(left.bbox.cy - right.bbox.cy)
            if manhattan < 250 or manhattan > 1400:
                continue
            midpoint = ((left.bbox.cx + right.bbox.cx) / 2.0, (left.bbox.cy + right.bbox.cy) / 2.0)
            label = rng.choice(labels)
            bbox = _bbox_for_label(label, midpoint[0], midpoint[1], rng).clamp_inside(bounds, padding=6.0)
            if _overlaps(scene, bbox, tolerance=8.0, ignore_ids={left.id, right.id}):
                continue
            node = _make_node(scene, label, bbox, metadata={"generated": True, "generated_keep": True})
            scene.add_edge(Edge(source=left.id, target=node.id, style="solid"))
            scene.add_edge(Edge(source=node.id, target=right.id, style="solid"))
            created += 1
            break
    return created


def apply_label_remix(scene: Scene, rng: random.Random, budget: int) -> int:
    if budget <= 0:
        return 0
    candidates = [node for node in scene.non_background_nodes() if node.label in {"general", "valve", "pump", "tank", "instrumentation"}]
    rng.shuffle(candidates)
    changed = 0
    for node in candidates:
        if changed >= budget:
            break
        new_label = node.label
        if node.label == "general":
            if node.bbox.area > 28000:
                new_label = rng.choice(["tank", "pump", "general"])
            else:
                new_label = rng.choice(["general", "valve", "instrumentation"])
        elif node.label == "valve":
            new_label = rng.choice(["valve", "general", "pump"])
        elif node.label == "instrumentation":
            new_label = rng.choice(["instrumentation", "general"])
        elif node.label == "pump":
            new_label = rng.choice(["pump", "general", "tank"])
        elif node.label == "tank":
            new_label = rng.choice(["tank", "general"])
        if new_label != node.label:
            node.label = new_label
            changed += 1
    return changed
