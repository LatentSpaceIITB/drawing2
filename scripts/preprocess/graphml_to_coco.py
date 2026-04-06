"""
graphml_to_coco.py
==================
Convert collapsed P&ID GraphML files + matching PNG images to a COCO-style
JSON annotation file suitable for Relationformer training/evaluation.

Expected input
--------------
A directory containing:
  *.graphml  -- collapsed GraphML (output of collapse_connectors.py)
  *.png      -- matching raster image (same stem as the GraphML)

The graphml files must be in the simplified collapsed format produced by
collapse_connectors.py (keys: label, xmin, ymin, xmax, ymax, edge_label).

Output JSON schema
------------------
{
  "info":       { ... },
  "categories": [ { "id": int, "name": str } ],
  "predicates": [ { "id": int, "name": str } ],
  "images":     [ { "id": int, "file_name": str, "height": int, "width": int } ],
  "annotations":[ {
      "id":          int,
      "image_id":    int,
      "category_id": int,
      "bbox":        [x, y, w, h],   # COCO format: top-left x,y + width, height
      "area":        float,
      "node_id":     str             # original GraphML node id (for relation linking)
  } ],
  "relations":  [ {
      "id":           int,
      "image_id":     int,
      "subject_id":   int,           # annotation id of subject node
      "object_id":    int,           # annotation id of object node
      "predicate_id": int
  } ]
}

Node categories (fixed vocabulary, 0-indexed internally, 1-indexed in COCO):
  1  valve
  2  pump
  3  instrumentation
  4  general
  5  tank
  6  arrow
  7  inlet/outlet

Predicate categories:
  1  solid
  2  non-solid

Usage
-----
  python scripts/preprocess/graphml_to_coco.py \\
      --graphml-dir outputs/preprocessed/real \\
      --image-dir   "data/PID2Graph OPEN100" \\
      --output      outputs/preprocessed/real_coco.json

  python scripts/preprocess/graphml_to_coco.py \\
      --graphml-dir outputs/preprocessed/synthetic \\
      --image-dir   outputs/open100_dense_1000_v2 \\
      --output      outputs/preprocessed/synthetic_coco.json

  # With image size auto-detection disabled (use sizes from GraphML bbox range)
  python scripts/preprocess/graphml_to_coco.py \\
      --graphml-dir outputs/preprocessed/real \\
      --image-dir   "data/PID2Graph OPEN100" \\
      --output      outputs/preprocessed/real_coco.json \\
      --no-read-image-size
"""

from __future__ import annotations

import argparse
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

NODE_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "valve"},
    {"id": 2, "name": "pump"},
    {"id": 3, "name": "instrumentation"},
    {"id": 4, "name": "general"},
    {"id": 5, "name": "tank"},
    {"id": 6, "name": "arrow"},
    {"id": 7, "name": "inlet/outlet"},
]

PREDICATE_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "solid"},
    {"id": 2, "name": "non-solid"},
]

LABEL_TO_CAT_ID: dict[str, int] = {c["name"]: c["id"] for c in NODE_CATEGORIES}
PREDICATE_TO_ID: dict[str, int] = {p["name"]: p["id"] for p in PREDICATE_CATEGORIES}

# ---------------------------------------------------------------------------
# GraphML parsing (collapsed format only)
# ---------------------------------------------------------------------------

NS = "http://graphml.graphdrawing.org/xmlns"


def _parse_collapsed_graphml(
    path: Path,
) -> tuple[
    list[dict[str, Any]],       # nodes: [{node_id, label, xmin, ymin, xmax, ymax}]
    list[dict[str, Any]],       # edges: [{src_id, tgt_id, edge_label}]
]:
    tree = ET.parse(path)
    root = tree.getroot()

    # Build key-id → attr-name map
    schema: dict[str, str] = {}
    for key_el in root.findall(f"{{{NS}}}key"):
        schema[key_el.get("id", "")] = key_el.get("attr.name", "")

    graph_el = root.find(f"{{{NS}}}graph")
    if graph_el is None:
        raise ValueError(f"No <graph> element in {path}")

    nodes: list[dict[str, Any]] = []
    edges_raw: list[dict[str, Any]] = []

    for node_el in graph_el.findall(f"{{{NS}}}node"):
        nid = node_el.get("id", "")
        data: dict[str, str] = {}
        for d in node_el.findall(f"{{{NS}}}data"):
            name = schema.get(d.get("key", ""), "")
            if name:
                data[name] = d.text or ""
        label = data.get("label", "")
        if label not in LABEL_TO_CAT_ID:
            # Skip any non-physical nodes that slipped through
            continue
        try:
            xmin = float(data["xmin"])
            ymin = float(data["ymin"])
            xmax = float(data["xmax"])
            ymax = float(data["ymax"])
        except (KeyError, ValueError) as e:
            log.warning("  Skipping node %s in %s: %s", nid, path.name, e)
            continue
        nodes.append({
            "node_id": nid,
            "label": label,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    for edge_el in graph_el.findall(f"{{{NS}}}edge"):
        src = edge_el.get("source", "")
        tgt = edge_el.get("target", "")
        data: dict[str, str] = {}
        for d in edge_el.findall(f"{{{NS}}}data"):
            name = schema.get(d.get("key", ""), "")
            if name:
                data[name] = d.text or ""
        edge_label = data.get("edge_label", "solid")
        edges_raw.append({"src_id": src, "tgt_id": tgt, "edge_label": edge_label})

    return nodes, edges_raw


# ---------------------------------------------------------------------------
# Image size detection
# ---------------------------------------------------------------------------

def _get_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) by reading just the PNG header."""
    import struct
    with open(image_path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG: {image_path}")
        f.read(4)  # chunk length
        chunk_type = f.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"Expected IHDR chunk in {image_path}")
        width = struct.unpack(">I", f.read(4))[0]
        height = struct.unpack(">I", f.read(4))[0]
    return width, height


def _infer_image_size_from_nodes(
    nodes: list[dict[str, Any]]
) -> tuple[int, int]:
    """Fallback: infer canvas size from max bounding box extent."""
    if not nodes:
        return 1, 1
    max_x = max(n["xmax"] for n in nodes)
    max_y = max(n["ymax"] for n in nodes)
    return int(max_x) + 1, int(max_y) + 1


# ---------------------------------------------------------------------------
# COCO builder
# ---------------------------------------------------------------------------

def build_coco(
    graphml_dir: Path,
    image_dir: Path,
    read_image_size: bool = True,
    graphml_pattern: str = "*.graphml",
) -> dict[str, Any]:

    graphml_files = sorted(graphml_dir.glob(graphml_pattern))
    if not graphml_files:
        raise FileNotFoundError(
            f"No files matching '{graphml_pattern}' in {graphml_dir}"
        )

    coco: dict[str, Any] = {
        "info": {
            "description": "P&ID graph extraction dataset (collapsed)",
            "graphml_dir": str(graphml_dir),
            "image_dir": str(image_dir),
        },
        "categories": NODE_CATEGORIES,
        "predicates": PREDICATE_CATEGORIES,
        "images": [],
        "annotations": [],
        "relations": [],
    }

    ann_id = 1
    rel_id = 1
    skipped_images = 0
    skipped_nodes = 0
    skipped_edges = 0

    for img_id, gml_path in enumerate(graphml_files, start=1):
        stem = gml_path.stem

        # Find matching image
        image_path = image_dir / (stem + ".png")
        if not image_path.exists():
            log.warning("  No PNG for %s — skipping", stem)
            skipped_images += 1
            continue

        # Parse GraphML
        try:
            nodes, edges_raw = _parse_collapsed_graphml(gml_path)
        except Exception as e:
            log.error("  Failed to parse %s: %s", gml_path.name, e)
            skipped_images += 1
            continue

        if not nodes:
            log.warning("  No physical nodes in %s — skipping", stem)
            skipped_images += 1
            continue

        # Image size
        if read_image_size:
            try:
                width, height = _get_image_size(image_path)
            except Exception as e:
                log.warning("  Could not read image size for %s: %s — inferring from nodes", stem, e)
                width, height = _infer_image_size_from_nodes(nodes)
        else:
            width, height = _infer_image_size_from_nodes(nodes)

        coco["images"].append({
            "id": img_id,
            "file_name": str(image_path),
            "stem": stem,
            "height": height,
            "width": width,
        })

        # Build node_id → annotation_id map (needed for edges)
        node_id_to_ann_id: dict[str, int] = {}

        for node in nodes:
            nid = node["node_id"]
            label = node["label"]
            cat_id = LABEL_TO_CAT_ID.get(label)
            if cat_id is None:
                log.debug("  Unknown label '%s' in %s — skipping node", label, stem)
                skipped_nodes += 1
                continue

            xmin = node["xmin"]
            ymin = node["ymin"]
            xmax = node["xmax"]
            ymax = node["ymax"]
            w = xmax - xmin
            h = ymax - ymin

            if w <= 0 or h <= 0:
                log.debug("  Degenerate bbox for node %s in %s — skipping", nid, stem)
                skipped_nodes += 1
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, w, h],   # COCO: [x, y, width, height]
                "area": w * h,
                "node_id": nid,               # original GraphML id for traceability
            })
            node_id_to_ann_id[nid] = ann_id
            ann_id += 1

        # Build relations
        for edge in edges_raw:
            src_nid = edge["src_id"]
            tgt_nid = edge["tgt_id"]
            elabel = edge["edge_label"]

            subj_ann = node_id_to_ann_id.get(src_nid)
            obj_ann = node_id_to_ann_id.get(tgt_nid)

            if subj_ann is None or obj_ann is None:
                # One or both endpoints were collapsed/missing — skip edge
                skipped_edges += 1
                continue

            pred_id = PREDICATE_TO_ID.get(elabel, 1)  # default solid

            coco["relations"].append({
                "id": rel_id,
                "image_id": img_id,
                "subject_id": subj_ann,
                "object_id": obj_ann,
                "predicate_id": pred_id,
            })
            rel_id += 1

        img_rel_count = sum(1 for r in coco["relations"] if r["image_id"] == img_id)
        log.info(
            "  %s  |  nodes=%d  edges=%d  image=%dx%d",
            stem, len(node_id_to_ann_id), img_rel_count, width, height
        )

    log.info(
        "Total: %d images, %d annotations, %d relations  "
        "[skipped: imgs=%d nodes=%d edges=%d]",
        len(coco["images"]),
        len(coco["annotations"]),
        len(coco["relations"]),
        skipped_images, skipped_nodes, skipped_edges,
    )
    return coco


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert collapsed P&ID GraphML files to COCO-style JSON."
    )
    parser.add_argument(
        "--graphml-dir", type=Path, required=True,
        help="Directory containing collapsed .graphml files"
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory containing .png images (must match graphml stems)"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output COCO JSON file path"
    )
    parser.add_argument(
        "--pattern", default="*.graphml",
        help="Glob pattern for graphml files (default: *.graphml)"
    )
    parser.add_argument(
        "--no-read-image-size", action="store_true",
        help="Infer image size from bounding boxes instead of reading PNG headers"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-node debug output"
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    coco = build_coco(
        graphml_dir=args.graphml_dir,
        image_dir=args.image_dir,
        read_image_size=not args.no_read_image_size,
        graphml_pattern=args.pattern,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    log.info("Saved COCO JSON → %s", args.output)


if __name__ == "__main__":
    main()
