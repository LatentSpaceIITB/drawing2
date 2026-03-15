from __future__ import annotations

import random

from .complexity import get_profile
from .scene import Edge, Node, Scene
from .template import GENERAL_CODES, INSTRUMENT_CODES, NOTE_TEMPLATES, PIPE_CODES, UNIT_CODES, VALVE_CODES


def _rand_code(rng: random.Random, choices: list[str], digits: int = 4) -> str:
    return f"{rng.choice(choices)}-{rng.randint(0, (10 ** digits) - 1):0{digits}d}"


def _pipe_spec(rng: random.Random) -> str:
    prefix = rng.choice(["2\"", "3\"", "5\"", "7\"x4\"", "8\"", "10\"", "12\"", "15\"", "16\""])
    return f"{prefix}-{rng.choice(PIPE_CODES)}-{rng.randint(1000, 9999)}"


def _instrument_text(rng: random.Random) -> str:
    return f"{rng.choice(INSTRUMENT_CODES)}\n{rng.randint(100, 999)}"


def _stacked_box_text(rng: random.Random) -> str:
    middle = rng.choice(["LG-10", "LC-10", "MI", "INS", "STA"])
    return f"{rng.randint(100, 999)}\n{middle}\n{rng.randint(100, 999)}"


def _equipment_text(rng: random.Random) -> str:
    prefix = rng.choice(UNIT_CODES)
    suffix = rng.choice(["VCT", "ION", "FL", "HX", "AN", "PU", "TK"])
    return f"{prefix}\n{suffix}-{rng.randint(110, 199)}{rng.choice(['A', 'B', ''])}"


def _station_text(rng: random.Random) -> str:
    return rng.choice(["STA", "INS", "MI", "NC", "NO"])


def _tank_text(rng: random.Random) -> str:
    return f"TK-{rng.randint(100, 999)}{rng.choice(['A', 'B', ''])}"


def _pump_text(rng: random.Random) -> str:
    return f"PU-{rng.randint(100, 999)}{rng.choice(['A', 'B', ''])}"


def _terminal_text(rng: random.Random) -> str:
    return rng.choice([_pipe_spec(rng), f"INLET {rng.randint(1, 9)}", f"OUTLET {rng.randint(1, 9)}"])


def _general_variant(node: Node, degree: int, rng: random.Random) -> str:
    area = node.bbox.area
    if area >= 48000:
        return rng.choice(["vessel", "unit_box", "pump_box"])
    if node.bbox.width >= 90 and node.bbox.height >= 70:
        return rng.choice(["stacked_box", "unit_box", "double_box"])
    if node.bbox.width >= 70 and degree <= 2:
        return rng.choice(["station_box", "reducer", "heater", "orifice"])
    return rng.choice(["station_box", "stacked_box", "heater", "tag_box", "terminal"])


def _instrument_variant(rng: random.Random) -> str:
    return rng.choices(["round", "panel", "inline", "dashed"], weights=[0.6, 0.15, 0.15, 0.1], k=1)[0]


def _valve_variant(rng: random.Random) -> str:
    return rng.choices(["gate", "control", "check", "relief"], weights=[0.45, 0.25, 0.2, 0.1], k=1)[0]


def _edge_should_get_label(edge: Edge, scene: Scene, rng: random.Random, rate: float) -> bool:
    source = scene.nodes[edge.source]
    target = scene.nodes[edge.target]
    dx = abs(source.bbox.cx - target.bbox.cx)
    dy = abs(source.bbox.cy - target.bbox.cy)
    length = dx + dy
    if edge.style == "non-solid":
        return length > 180 and rng.random() < min(0.5, rate + 0.05)
    return length > 220 and rng.random() < rate


def annotate_scene(scene: Scene, seed: int = 1401, profile_name: str = "medium") -> None:
    rng = random.Random(seed)
    profile = get_profile(profile_name)
    adjacency = scene.adjacency()

    notes = NOTE_TEMPLATES[:]
    rng.shuffle(notes)
    scene.metadata.update(
        {
            "project_name": "SAMPLE Project",
            "drawing_title": "SYNTHETIC PROCESS FLOW DIAGRAM",
            "scheme_title": "HYBRID BOOTSTRAP FLOW SCHEME",
            "sheet_name": f"SAMPLE_{rng.randint(1000, 9999)}.PNG",
            "project_location": _rand_code(rng, ["IPY", "OTX", "AI", "ZX", "CVCS"], digits=5),
            "organization": rng.choice(["P-0", "T-0", "M-2", "B-1"]),
            "contractor_project": _rand_code(rng, ["ZX", "AI", "CV", "PR"], digits=4),
            "contractor_no": str(rng.randint(800, 999)),
            "unit": str(rng.randint(10, 99)) + "-" + str(rng.randint(1000, 9999)),
            "scale": rng.choice(["Std.", "None", "1:100", "1:50"]),
            "drawing_sheet": str(rng.randint(1000000, 9999999)),
            "revision": str(rng.randint(1, 5)),
            "notes": notes[: profile.note_count],
            "revision_rows": [
                (
                    label,
                    f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2{rng.randint(2, 6)}",
                    rng.choice(["IT", "QA", "JD", "XP"]),
                    rng.choice(["S/W", "ML", "ZJ", "TA"]),
                )
                for label in ["A", "B", "C", "D"]
            ],
            "source_dataset": scene.metadata.get("source_dataset", "unknown"),
            "generation_profile": profile.name,
        }
    )

    for node in scene.sorted_nodes():
        degree = len(adjacency.get(node.id, []))
        node.metadata["degree"] = degree
        if node.label == "instrumentation":
            node.metadata["variant"] = _instrument_variant(rng)
            node.metadata["text"] = _instrument_text(rng)
        elif node.label == "valve":
            node.metadata["variant"] = _valve_variant(rng)
            node.metadata["tag"] = _rand_code(rng, VALVE_CODES)
        elif node.label == "general":
            variant = _general_variant(node, degree, rng)
            node.metadata["variant"] = variant
            if variant == "vessel":
                node.metadata["text"] = _equipment_text(rng)
                node.metadata["caption"] = rng.choice(["VESSEL", "ACCUMULATOR", "TANK"])
            elif variant == "unit_box":
                node.metadata["text"] = _equipment_text(rng).replace("\n", "-")
            elif variant == "pump_box":
                node.metadata["text"] = f"{rng.choice(UNIT_CODES)}-PU-{rng.randint(110, 199)}"
            elif variant == "double_box":
                node.metadata["text"] = f"{rng.randint(100, 999)}\n{rng.choice(['LG-10', 'LC-10', 'STA'])}"
            elif variant == "stacked_box":
                node.metadata["text"] = _stacked_box_text(rng)
            elif variant == "heater":
                node.metadata["text"] = f"{rng.choice(GENERAL_CODES)}-{rng.randint(10, 99)}"
            elif variant == "reducer":
                node.metadata["text"] = _pipe_spec(rng)
            elif variant == "orifice":
                node.metadata["text"] = _pipe_spec(rng)
            elif variant == "tag_box":
                node.metadata["text"] = _rand_code(rng, GENERAL_CODES, digits=2)
            elif variant == "terminal":
                node.metadata["text"] = rng.choice(["NC", "NO", "1/2\"", "2\""])
            else:
                node.metadata["text"] = _station_text(rng)
        elif node.label == "arrow":
            node.metadata["variant"] = rng.choice(["filled", "outline"])
        elif node.label == "tank":
            node.metadata["variant"] = rng.choice(["vertical_tank", "horizontal_tank"])
            node.metadata["text"] = _tank_text(rng)
            node.metadata["caption"] = rng.choice(["TANK", "VESSEL", "DRUM"])
        elif node.label == "pump":
            node.metadata["variant"] = rng.choice(["inline_pump", "centrifugal"])
            node.metadata["text"] = _pump_text(rng)
        elif node.label == "inlet/outlet":
            node.metadata["variant"] = rng.choice(["inlet", "outlet", "terminal_box"])
            node.metadata["text"] = _terminal_text(rng)

    for edge in scene.edges:
        if _edge_should_get_label(edge, scene, rng, profile.edge_label_rate):
            edge.metadata["text"] = _pipe_spec(rng)
        elif edge.style == "non-solid" and rng.random() < 0.18:
            edge.metadata["text"] = f"INS ({rng.randint(35, 90)}C)"
