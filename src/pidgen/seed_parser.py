from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from .scene import BoundingBox, Edge, Node, Scene


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def _first_float(values: list[str]) -> float:
    for value in values:
        if value is None:
            continue
        return float(value)
    raise ValueError("missing numeric GraphML value")


def _infer_source_dataset(graphml_path: Path) -> str:
    path_text = str(graphml_path)
    if "PID2Graph OPEN100" in path_text:
        return "open100"
    if "Dataset PID" in path_text:
        return "dataset_pid"
    return "unknown"


def _anchor_side(bbox: BoundingBox, page_bbox: BoundingBox) -> str:
    distances = {
        "left": abs(bbox.xmin - page_bbox.xmin),
        "right": abs(page_bbox.xmax - bbox.xmax),
        "top": abs(bbox.ymin - page_bbox.ymin),
        "bottom": abs(page_bbox.ymax - bbox.ymax),
    }
    return min(distances.items(), key=lambda item: item[1])[0]


def parse_graphml(graphml_path: str | Path) -> Scene:
    graphml_path = Path(graphml_path)
    source_dataset = _infer_source_dataset(graphml_path)
    root = ET.parse(graphml_path).getroot()

    key_names = {
        key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"])
        for key in root.findall("g:key", GRAPHML_NS)
    }

    nodes: dict[str, Node] = {}
    max_x = 0.0
    max_y = 0.0
    raw_nodes: list[tuple[str, str, BoundingBox]] = []
    for xml_node in root.findall(".//g:node", GRAPHML_NS):
        values: defaultdict[str, list[str]] = defaultdict(list)
        for data in xml_node.findall("g:data", GRAPHML_NS):
            values[key_names.get(data.attrib["key"], data.attrib["key"])].append(data.text or "")
        label = (values.get("label", [""])[0] or "").strip()
        bbox = BoundingBox(
            _first_float(values["xmin"]),
            _first_float(values["ymin"]),
            _first_float(values["xmax"]),
            _first_float(values["ymax"]),
        )
        max_x = max(max_x, bbox.xmax)
        max_y = max(max_y, bbox.ymax)
        raw_nodes.append((xml_node.attrib["id"], label, bbox))

    page_bbox = BoundingBox(0.0, 0.0, max_x, max_y)
    content_nodes = [bbox for _, label, bbox in raw_nodes if label != "background"]
    if content_nodes:
        content_bbox = BoundingBox(
            min(bbox.xmin for bbox in content_nodes),
            min(bbox.ymin for bbox in content_nodes),
            max(bbox.xmax for bbox in content_nodes),
            max(bbox.ymax for bbox in content_nodes),
        )
    else:
        content_bbox = page_bbox

    for node_id, label, bbox in raw_nodes:
        metadata: dict[str, object] = {
            "original_label": label,
            "source_bbox": bbox.to_int_tuple(),
            "source_dataset": source_dataset,
        }
        if label == "inlet/outlet":
            metadata["anchor_side"] = _anchor_side(bbox, page_bbox)
        nodes[node_id] = Node(
            id=node_id,
            label=label,
            bbox=bbox,
            metadata=metadata,
        )

    edges: list[Edge] = []
    for xml_edge in root.findall(".//g:edge", GRAPHML_NS):
        style = "solid"
        for data in xml_edge.findall("g:data", GRAPHML_NS):
            if key_names.get(data.attrib["key"]) == "edge_label":
                style = (data.text or "solid").strip() or "solid"
                break
        edges.append(
            Edge(
                source=xml_edge.attrib["source"],
                target=xml_edge.attrib["target"],
                style=style,
            )
        )

    return Scene(
        width=int(round(max_x)),
        height=int(round(max_y)),
        nodes=nodes,
        edges=edges,
        metadata={
            "seed_path": str(graphml_path),
            "seed_name": graphml_path.stem,
            "source_dataset": source_dataset,
            "source_page_bbox": page_bbox.to_int_tuple(),
            "source_content_bbox": content_bbox.to_int_tuple(),
        },
    )
