"""
collapse_connectors.py
======================
Collapse connector and crossing nodes from P&ID GraphML files.

What this does
--------------
- Removes `connector` nodes (pipe routing waypoints, ~8x8 px)
- Removes `crossing` nodes (pipe-over-pipe non-connections, ~8x8 px)
- Removes `background` nodes (page border rectangles)
- Connects physical components that were bridged by connector chains
- Inherits edge labels by majority vote along each connector chain
- Normalises bounding-box coordinates to canonical (xmin, ymin, xmax, ymax) floats

Handles both GraphML formats:
  Real OPEN-100:    primary nodes use d1(xmin), d2(xmax), d3(ymin), d4(ymax) as doubles
                    connector/crossing use d5(xmin), d6(ymin), d7(xmax), d8(ymax) as longs
  Synthetic gen:    primary nodes use d1(xmin), d2(ymin), d3(xmax), d4(ymax) as longs
                    other nodes use d5(xmin), d6(ymin), d7(xmax), d8(ymax) as doubles

Output
------
GraphML file with only physical component nodes and their direct connections.
Each node has:
  <data key="label">   -- class label
  <data key="xmin">    -- float
  <data key="ymin">    -- float
  <data key="xmax">    -- float
  <data key="ymax">    -- float
Each edge has:
  <data key="edge_label">  -- "solid" or "non-solid"

Usage
-----
  # Single file
  python scripts/preprocess/collapse_connectors.py \\
      --input  "data/PID2Graph OPEN100/1.graphml" \\
      --output outputs/preprocessed/real/1.graphml

  # Batch: all real data
  python scripts/preprocess/collapse_connectors.py \\
      --input-dir  "data/PID2Graph OPEN100" \\
      --output-dir outputs/preprocessed/real

  # Batch: all synthetic data
  python scripts/preprocess/collapse_connectors.py \\
      --input-dir  outputs/open100_dense_1000_v2 \\
      --output-dir outputs/preprocessed/synthetic \\
      --pattern "*.graphml"
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node classes that are physical components (kept after collapsing)
# ---------------------------------------------------------------------------
PHYSICAL_LABELS = {
    "valve",
    "pump",
    "instrumentation",
    "general",
    "tank",
    "arrow",
    "inlet/outlet",
}

# Node classes that are removed entirely
REMOVE_LABELS = {"connector", "crossing", "background"}

# GraphML namespace
NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NodeData(NamedTuple):
    node_id: str
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


# ---------------------------------------------------------------------------
# GraphML parsing
# ---------------------------------------------------------------------------

def _parse_key_schema(root: ET.Element) -> dict[str, str]:
    """Return mapping of key id -> attr.name for node keys."""
    schema: dict[str, str] = {}
    for key_el in root.findall(f"{{{NS}}}key"):
        kid = key_el.get("id", "")
        name = key_el.get("attr.name", "")
        schema[kid] = name
    return schema


def _get_data(node_el: ET.Element, schema: dict[str, str]) -> dict[str, str]:
    """Return {attr_name: value} for all <data> children of a node element."""
    result: dict[str, str] = {}
    for data_el in node_el.findall(f"{{{NS}}}data"):
        key = data_el.get("key", "")
        name = schema.get(key, "")
        if name:
            result[name] = data_el.text or ""
    return result


def _resolve_bbox(data: dict[str, str]) -> tuple[float, float, float, float] | None:
    """
    Extract (xmin, ymin, xmax, ymax) from a node's data dict.

    Handles the two different GraphML conventions:

    Real OPEN-100 (primary nodes):
      d1=xmin, d2=xmax, d3=ymin, d4=ymax  (doubles)
      → attr names: xmin, xmax, ymin, ymax

    Synthetic gen (primary nodes):
      d1=xmin, d2=ymin, d3=xmax, d4=ymax  (longs)
      → attr names: xmin, ymin, xmax, ymax

    Both cases have the same attr.name values; we rely on those names, not key IDs.
    The difference is xmax/ymin ordering which the attr names resolve unambiguously.
    """
    try:
        xmin = float(data["xmin"])
        ymin = float(data["ymin"])
        xmax = float(data["xmax"])
        ymax = float(data["ymax"])
        return xmin, ymin, xmax, ymax
    except (KeyError, ValueError):
        return None


def parse_graphml(path: Path) -> tuple[dict[str, NodeData], dict[tuple[str, str], str]]:
    """
    Parse a GraphML file into:
      nodes:  {node_id: NodeData}
      edges:  {(src_id, tgt_id): edge_label}   (undirected, src < tgt by string sort)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    schema = _parse_key_schema(root)

    nodes: dict[str, NodeData] = {}
    edges: dict[tuple[str, str], str] = {}

    graph_el = root.find(f"{{{NS}}}graph")
    if graph_el is None:
        raise ValueError(f"No <graph> element found in {path}")

    # Parse nodes
    for node_el in graph_el.findall(f"{{{NS}}}node"):
        nid = node_el.get("id", "")
        data = _get_data(node_el, schema)
        label = data.get("label", "")
        bbox = _resolve_bbox(data)
        if bbox is None:
            # Some nodes (background, connector) may use the secondary key set.
            # Try reading whatever coordinate values exist.
            bbox = (0.0, 0.0, 0.0, 0.0)
        nodes[nid] = NodeData(
            node_id=nid,
            label=label,
            xmin=bbox[0],
            ymin=bbox[1],
            xmax=bbox[2],
            ymax=bbox[3],
        )

    # Parse edges  (undirected: store with sorted key)
    for edge_el in graph_el.findall(f"{{{NS}}}edge"):
        src = edge_el.get("source", "")
        tgt = edge_el.get("target", "")
        data = _get_data(edge_el, schema)
        label = data.get("edge_label", "solid")
        key = (min(src, tgt), max(src, tgt))
        edges[key] = label

    return nodes, edges


# ---------------------------------------------------------------------------
# Collapse algorithm
# ---------------------------------------------------------------------------

def collapse_connectors(
    nodes: dict[str, NodeData],
    edges: dict[tuple[str, str], str],
) -> tuple[dict[str, NodeData], dict[tuple[str, str], str]]:
    """
    Remove connector, crossing, and background nodes.
    Connect physical components that were bridged via connector chains.

    Algorithm:
      1. Remove crossing nodes and all their incident edges entirely.
      2. Build an adjacency list over remaining nodes.
      3. For each physical node, BFS/DFS through connector-only paths to find
         all reachable physical nodes. Add a direct edge for each pair with
         majority-vote edge label from the traversed path edges.
      4. Remove all connector nodes.
      5. Remove all background nodes.
    """

    # Step 1: identify which nodes to remove completely (crossings + background)
    remove_completely = {
        nid for nid, nd in nodes.items()
        if nd.label in ("crossing", "background")
    }

    # Step 2: build adjacency list excluding removed nodes
    adj: dict[str, dict[str, str]] = defaultdict(dict)
    for (src, tgt), elabel in edges.items():
        if src in remove_completely or tgt in remove_completely:
            continue
        adj[src][tgt] = elabel
        adj[tgt][src] = elabel

    # Step 3: find physical nodes reachable through connector-only paths
    physical_ids = {
        nid for nid, nd in nodes.items()
        if nd.label in PHYSICAL_LABELS
    }
    connector_ids = {
        nid for nid, nd in nodes.items()
        if nd.label == "connector"
    }

    new_edges: dict[tuple[str, str], list[str]] = defaultdict(list)

    visited_from: set[str] = set()

    for start_phys in physical_ids:
        if start_phys not in adj:
            continue
        # BFS from this physical node, traversing only through connectors
        queue: list[tuple[str, list[str]]] = []
        # Each queue item: (current_node, edge_labels_on_path_so_far)
        for neighbor, elabel in adj[start_phys].items():
            if neighbor in connector_ids:
                queue.append((neighbor, [elabel]))
            elif neighbor in physical_ids:
                # Direct physical-to-physical edge (already exists, no connector)
                key = (min(start_phys, neighbor), max(start_phys, neighbor))
                new_edges[key].append(elabel)

        visited_connector: set[str] = set()
        while queue:
            current, path_labels = queue.pop(0)
            if current in visited_connector:
                continue
            visited_connector.add(current)

            for neighbor, elabel in adj[current].items():
                accumulated = path_labels + [elabel]
                if neighbor == start_phys:
                    # Loop back — skip
                    continue
                elif neighbor in physical_ids:
                    # Found a physical component at the end of the connector chain
                    key = (min(start_phys, neighbor), max(start_phys, neighbor))
                    new_edges[key].extend(accumulated)
                elif neighbor in connector_ids and neighbor not in visited_connector:
                    queue.append((neighbor, accumulated))

    # Step 4: compute majority-vote edge label for each new edge
    collapsed_edges: dict[tuple[str, str], str] = {}
    for key, label_list in new_edges.items():
        if not label_list:
            collapsed_edges[key] = "solid"
            continue
        counts = Counter(label_list)
        # Majority vote; tie-break to "solid"
        majority = counts.most_common(1)[0][0]
        # If it's genuinely tied, default to solid
        if len(counts) == 2 and counts["solid"] == counts["non-solid"]:
            majority = "solid"
        collapsed_edges[key] = majority

    # Step 5: keep only physical nodes
    collapsed_nodes = {nid: nd for nid, nd in nodes.items() if nid in physical_ids}

    return collapsed_nodes, collapsed_edges


# ---------------------------------------------------------------------------
# GraphML export (clean simplified format)
# ---------------------------------------------------------------------------

_GRAPHML_HEADER = """\
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
           http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="label"      for="node" attr.name="label"      attr.type="string" />
  <key id="xmin"       for="node" attr.name="xmin"       attr.type="double" />
  <key id="ymin"       for="node" attr.name="ymin"       attr.type="double" />
  <key id="xmax"       for="node" attr.name="xmax"       attr.type="double" />
  <key id="ymax"       for="node" attr.name="ymax"       attr.type="double" />
  <key id="edge_label" for="edge" attr.name="edge_label" attr.type="string" />
  <graph edgedefault="undirected">
"""

_GRAPHML_FOOTER = """\
  </graph>
</graphml>
"""


def write_collapsed_graphml_safe(
    path: Path,
    nodes: dict[str, NodeData],
    edges: dict[tuple[str, str], str],
) -> None:
    """Version that uses the local `edges` parameter (not the module-level name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_GRAPHML_HEADER]

    for nd in sorted(nodes.values(), key=lambda n: n.node_id):
        lines.append(f'    <node id="{nd.node_id}">\n')
        lines.append(f'      <data key="label">{nd.label}</data>\n')
        lines.append(f'      <data key="xmin">{nd.xmin:.4f}</data>\n')
        lines.append(f'      <data key="ymin">{nd.ymin:.4f}</data>\n')
        lines.append(f'      <data key="xmax">{nd.xmax:.4f}</data>\n')
        lines.append(f'      <data key="ymax">{nd.ymax:.4f}</data>\n')
        lines.append(f'    </node>\n')

    edge_id = 0
    for (src, tgt), elabel in sorted(edges.items()):
        lines.append(f'    <edge id="e{edge_id}" source="{src}" target="{tgt}">\n')
        lines.append(f'      <data key="edge_label">{elabel}</data>\n')
        lines.append(f'    </edge>\n')
        edge_id += 1

    lines.append(_GRAPHML_FOOTER)
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def print_stats(
    source: Path,
    orig_nodes: dict[str, NodeData],
    orig_edges: dict[tuple[str, str], str],
    coll_nodes: dict[str, NodeData],
    coll_edges: dict[tuple[str, str], str],
) -> None:
    orig_by_label: Counter[str] = Counter(nd.label for nd in orig_nodes.values())
    coll_by_label: Counter[str] = Counter(nd.label for nd in coll_nodes.values())
    log.info(
        "%s  |  nodes %d→%d  edges %d→%d  "
        "[removed: conn=%d cross=%d bg=%d]",
        source.name,
        len(orig_nodes), len(coll_nodes),
        len(orig_edges), len(coll_edges),
        orig_by_label.get("connector", 0),
        orig_by_label.get("crossing", 0),
        orig_by_label.get("background", 0),
    )
    log.debug("  collapsed node classes: %s", dict(coll_by_label))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path, stats: bool = True) -> None:
    orig_nodes, orig_edges = parse_graphml(input_path)
    coll_nodes, coll_edges = collapse_connectors(orig_nodes, orig_edges)
    write_collapsed_graphml_safe(output_path, coll_nodes, coll_edges)  # noqa: E501
    if stats:
        print_stats(input_path, orig_nodes, orig_edges, coll_nodes, coll_edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse connector/crossing nodes from P&ID GraphML files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single input .graphml file")
    group.add_argument("--input-dir", type=Path, help="Directory of .graphml files")

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--output", type=Path, help="Single output .graphml file")
    out_group.add_argument("--output-dir", type=Path, help="Output directory")

    parser.add_argument(
        "--pattern", default="*.graphml",
        help="Glob pattern for input files when using --input-dir (default: *.graphml)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-class node counts after collapsing"
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.input:
        # Single file mode
        input_path = args.input
        if args.output:
            output_path = args.output
        elif args.output_dir:
            output_path = args.output_dir / input_path.name
        else:
            output_path = input_path.parent / (input_path.stem + "_collapsed.graphml")
        process_file(input_path, output_path)

    else:
        # Batch mode
        input_dir = args.input_dir
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = input_dir.parent / (input_dir.name + "_collapsed")

        files = sorted(input_dir.glob(args.pattern))
        if not files:
            log.error("No files matching '%s' in %s", args.pattern, input_dir)
            return

        log.info("Processing %d files from %s → %s", len(files), input_dir, output_dir)
        success = 0
        for f in files:
            out = output_dir / f.name
            try:
                process_file(f, out)
                success += 1
            except Exception as exc:
                log.error("FAILED %s: %s", f.name, exc)

        log.info("Done: %d/%d files processed successfully.", success, len(files))


if __name__ == "__main__":
    main()
