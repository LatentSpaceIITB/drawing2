from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pidgen import generate_outputs


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def main() -> None:
    output_dir = ROOT / "outputs" / "verify"
    manifest = generate_outputs(
        seed_graphml=ROOT / "data" / "Dataset PID" / "0.graphml",
        output_dir=output_dir,
        stem="verify_phase1",
        seed=1401,
    )

    manifest_path = Path(str(manifest["manifest"]))
    graphml_path = Path(str(manifest["outputs"]["graphml"]))
    png_path = Path(str(manifest["outputs"]["png"]))
    svg_path = Path(str(manifest["outputs"]["svg"]))

    assert manifest_path.exists(), "manifest missing"
    assert graphml_path.exists(), "graphml missing"
    assert png_path.exists(), "png missing"
    assert svg_path.exists(), "svg missing"

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = ET.parse(graphml_path).getroot()
    node_count = len(root.findall(".//g:node", GRAPHML_NS))
    edge_count = len(root.findall(".//g:edge", GRAPHML_NS))
    assert node_count == int(manifest_data["scene_summary"]["node_count"]), "node count mismatch"
    assert edge_count == int(manifest_data["scene_summary"]["edge_count"]), "edge count mismatch"
    print("verification passed")
    print(json.dumps(manifest_data["scene_summary"], indent=2))


if __name__ == "__main__":
    main()
