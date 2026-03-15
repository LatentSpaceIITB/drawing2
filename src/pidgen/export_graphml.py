from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .scene import Scene


GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", GRAPHML_NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def _node_data(parent: ET.Element, key: str, value: str) -> None:
    element = ET.SubElement(parent, f"{{{GRAPHML_NS}}}data", {"key": key})
    element.text = value


def export_graphml(scene: Scene, output_path: str | Path) -> None:
    output_path = Path(output_path)
    root = ET.Element(
        f"{{{GRAPHML_NS}}}graphml",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{GRAPHML_NS} {GRAPHML_NS}/1.0/graphml.xsd",
        },
    )

    keys = [
        ("d9", "edge", "edge_label", "string"),
        ("d8", "node", "ymax", "double"),
        ("d7", "node", "xmax", "double"),
        ("d6", "node", "ymin", "double"),
        ("d5", "node", "xmin", "double"),
        ("d4", "node", "ymax", "long"),
        ("d3", "node", "xmax", "long"),
        ("d2", "node", "ymin", "long"),
        ("d1", "node", "xmin", "long"),
        ("d0", "node", "label", "string"),
    ]
    for key_id, key_for, attr_name, attr_type in keys:
        ET.SubElement(
            root,
            f"{{{GRAPHML_NS}}}key",
            {
                "id": key_id,
                "for": key_for,
                "attr.name": attr_name,
                "attr.type": attr_type,
            },
        )

    graph = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph", {"edgedefault": "undirected"})
    for node in scene.sorted_nodes():
        node_el = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node", {"id": node.id})
        bbox = node.bbox.to_int_tuple()
        _node_data(node_el, "d0", node.label)
        if node.label == "background":
            _node_data(node_el, "d5", f"{float(bbox[0]):.1f}")
            _node_data(node_el, "d6", f"{float(bbox[1]):.1f}")
            _node_data(node_el, "d7", f"{float(bbox[2]):.1f}")
            _node_data(node_el, "d8", f"{float(bbox[3]):.1f}")
        else:
            _node_data(node_el, "d1", str(bbox[0]))
            _node_data(node_el, "d2", str(bbox[1]))
            _node_data(node_el, "d3", str(bbox[2]))
            _node_data(node_el, "d4", str(bbox[3]))

    for edge in scene.edges:
        edge_el = ET.SubElement(
            graph,
            f"{{{GRAPHML_NS}}}edge",
            {"source": edge.source, "target": edge.target},
        )
        _node_data(edge_el, "d9", edge.style)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
