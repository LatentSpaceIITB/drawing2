from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        self.xmin, self.xmax = sorted((self.xmin, self.xmax))
        self.ymin, self.ymax = sorted((self.ymin, self.ymax))

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return self.xmin + (self.width / 2.0)

    @property
    def cy(self) -> float:
        return self.ymin + (self.height / 2.0)

    def translate(self, dx: float, dy: float) -> "BoundingBox":
        return BoundingBox(
            self.xmin + dx,
            self.ymin + dy,
            self.xmax + dx,
            self.ymax + dy,
        )

    def centered_at(self, cx: float, cy: float) -> "BoundingBox":
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return BoundingBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def scale_about_center(self, sx: float, sy: float | None = None) -> "BoundingBox":
        sy = sx if sy is None else sy
        half_w = (self.width * sx) / 2.0
        half_h = (self.height * sy) / 2.0
        return BoundingBox(self.cx - half_w, self.cy - half_h, self.cx + half_w, self.cy + half_h)

    def clamp_inside(self, bounds: "BoundingBox", padding: float = 0.0) -> "BoundingBox":
        min_x = bounds.xmin + padding
        max_x = bounds.xmax - padding
        min_y = bounds.ymin + padding
        max_y = bounds.ymax - padding

        dx = 0.0
        dy = 0.0
        if self.xmin < min_x:
            dx = min_x - self.xmin
        elif self.xmax > max_x:
            dx = max_x - self.xmax
        if self.ymin < min_y:
            dy = min_y - self.ymin
        elif self.ymax > max_y:
            dy = max_y - self.ymax
        return self.translate(dx, dy)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.xmin)),
            int(round(self.ymin)),
            int(round(self.xmax)),
            int(round(self.ymax)),
        )

    @classmethod
    def from_points(cls, points: Iterable[tuple[float, float]]) -> "BoundingBox":
        pts = list(points)
        if not pts:
            raise ValueError("cannot build BoundingBox from empty point set")
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        return cls(min(xs), min(ys), max(xs), max(ys))


@dataclass
class Node:
    id: str
    label: str
    bbox: BoundingBox
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    style: str = "solid"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Scene:
    width: int
    height: int
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def clone(self) -> "Scene":
        return deepcopy(self)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def sorted_nodes(self) -> list[Node]:
        return [self.nodes[node_id] for node_id in sorted(self.nodes)]

    def nodes_by_label(self, label: str) -> list[Node]:
        return [node for node in self.nodes.values() if node.label == label]

    def non_background_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.label != "background"]

    def content_bbox(self) -> BoundingBox:
        nodes = self.non_background_nodes()
        if not nodes:
            return BoundingBox(0.0, 0.0, float(self.width), float(self.height))
        return BoundingBox(
            min(node.bbox.xmin for node in nodes),
            min(node.bbox.ymin for node in nodes),
            max(node.bbox.xmax for node in nodes),
            max(node.bbox.ymax for node in nodes),
        )

    def adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
            adjacency.setdefault(edge.target, []).append(edge.source)
        return adjacency

    def degree(self, node_id: str) -> int:
        return len(self.adjacency().get(node_id, []))
