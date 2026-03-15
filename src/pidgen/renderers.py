from __future__ import annotations

from math import atan2, cos, dist, pi, sin
from pathlib import Path
from typing import Sequence, cast
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

from .scene import BoundingBox, Edge, Node, Scene
from .template import (
    FONT_FAMILY,
    FRAME_BOX,
    INK_RGB,
    NOTES_BOX,
    PAPER_RGB,
    REVISION_BOX,
    RIGHT_PANEL_BOX,
    SVG_INK,
    SVG_PAPER,
    TITLE_BLOCK_BOX,
    WHITE_RGB,
)


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FontLike = ImageFont.FreeTypeFont | ImageFont.ImageFont


def _load_font(size: int) -> FontLike:
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        return ImageFont.load_default()


def _fonts() -> dict[str, FontLike]:
    return {
        "tiny": _load_font(16),
        "small": _load_font(20),
        "medium": _load_font(26),
        "large": _load_font(34),
    }


def _node_center(node: Node) -> tuple[float, float]:
    return node.bbox.cx, node.bbox.cy


def _route_edge(scene: Scene, edge: Edge) -> list[tuple[float, float]]:
    source = scene.nodes[edge.source]
    target = scene.nodes[edge.target]
    sx, sy = _node_center(source)
    tx, ty = _node_center(target)
    if abs(sx - tx) < 1 or abs(sy - ty) < 1:
        return [(sx, sy), (tx, ty)]
    if abs(sx - tx) >= abs(sy - ty):
        return [(sx, sy), (tx, sy), (tx, ty)]
    return [(sx, sy), (sx, ty), (tx, ty)]


def _polyline_midpoint(points: list[tuple[float, float]]) -> tuple[float, float]:
    segments = []
    total = 0.0
    if len(points) == 1:
        return points[0]
    for start, end in zip(points, points[1:]):
        length = dist(start, end)
        total += length
        segments.append((start, end, length))
    if total == 0:
        return points[0]
    cursor = total / 2.0
    for start, end, length in segments:
        if cursor <= length:
            ratio = cursor / max(length, 1.0)
            return (
                start[0] + ((end[0] - start[0]) * ratio),
                start[1] + ((end[1] - start[1]) * ratio),
            )
        cursor -= length
    return points[-1]


def _polyline_orientation(points: Sequence[tuple[float, float]]) -> str:
    horizontal = 0.0
    vertical = 0.0
    for start, end in zip(points, points[1:]):
        if abs(start[0] - end[0]) >= abs(start[1] - end[1]):
            horizontal += dist(start, end)
        else:
            vertical += dist(start, end)
    return "vertical" if vertical > horizontal else "horizontal"


def _edge_direction(scene: Scene, node: Node) -> tuple[float, float]:
    adjacency = scene.adjacency()
    neighbors = adjacency.get(node.id, [])
    if not neighbors:
        return (1.0, 0.0)
    other = scene.nodes[neighbors[0]]
    dx = node.bbox.cx - other.bbox.cx
    dy = node.bbox.cy - other.bbox.cy
    if abs(dx) >= abs(dy):
        return (1.0 if dx >= 0 else -1.0, 0.0)
    return (0.0, 1.0 if dy >= 0 else -1.0)


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    dash: int = 20,
    gap: int = 12,
    width: int = 2,
) -> None:
    x1, y1 = start
    x2, y2 = end
    segment_length = dist(start, end)
    if segment_length == 0:
        return
    dx = (x2 - x1) / segment_length
    dy = (y2 - y1) / segment_length
    cursor = 0.0
    while cursor < segment_length:
        dash_end = min(cursor + dash, segment_length)
        sx = x1 + (dx * cursor)
        sy = y1 + (dy * cursor)
        ex = x1 + (dx * dash_end)
        ey = y1 + (dy * dash_end)
        draw.line((sx, sy, ex, ey), fill=INK_RGB, width=width)
        cursor += dash + gap


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    dashed: bool = False,
    width: int = 2,
) -> None:
    for start, end in zip(points, points[1:]):
        if dashed:
            _draw_dashed_segment(draw, start, end, width=width)
        else:
            draw.line((start[0], start[1], end[0], end[1]), fill=INK_RGB, width=width)


def _draw_dashed_box(draw: ImageDraw.ImageDraw, bbox: BoundingBox) -> None:
    x1, y1, x2, y2 = bbox.to_int_tuple()
    _draw_dashed_segment(draw, (x1, y1), (x2, y1), dash=18, gap=10, width=2)
    _draw_dashed_segment(draw, (x2, y1), (x2, y2), dash=18, gap=10, width=2)
    _draw_dashed_segment(draw, (x2, y2), (x1, y2), dash=18, gap=10, width=2)
    _draw_dashed_segment(draw, (x1, y2), (x1, y1), dash=18, gap=10, width=2)


def _multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: FontLike,
    anchor: str = "mm",
) -> None:
    draw.multiline_text(xy, text, fill=INK_RGB, font=font, anchor=anchor, align="center", spacing=1)


def _draw_sheet_template(draw: ImageDraw.ImageDraw, scene: Scene, fonts: dict[str, FontLike]) -> None:
    draw.rectangle((0, 0, scene.width, scene.height), fill=PAPER_RGB)
    draw.rectangle(RIGHT_PANEL_BOX.to_int_tuple(), fill=PAPER_RGB)
    _draw_dashed_box(draw, FRAME_BOX)

    draw.rectangle(NOTES_BOX.to_int_tuple(), outline=INK_RGB, width=2)
    draw.rectangle(REVISION_BOX.to_int_tuple(), outline=INK_RGB, width=2)
    draw.rectangle(TITLE_BLOCK_BOX.to_int_tuple(), outline=INK_RGB, width=2)

    draw.text((6215, 258), "NOTES", fill=INK_RGB, font=fonts["medium"], anchor="ma")
    y = 310
    notes_value = scene.metadata.get("notes", [])
    notes = notes_value if isinstance(notes_value, list) else []
    for index, note in enumerate(notes, start=1):
        draw.text((5660, y), f"{index}.", fill=INK_RGB, font=fonts["tiny"], anchor="la")
        draw.multiline_text((5715, y), str(note), fill=INK_RGB, font=fonts["tiny"], spacing=1)
        y += 66

    x1, y1, x2, _ = REVISION_BOX.to_int_tuple()
    row_height = 58
    for offset in range(1, 5):
        draw.line((x1, y1 + offset * row_height, x2, y1 + offset * row_height), fill=INK_RGB, width=1)
    for xpos in (5735, 5925, 6115, 6285, 6465):
        draw.line((xpos, y1, xpos, y1 + (row_height * 5)), fill=INK_RGB, width=1)
    headers = ["ISSUE", "DATE", "MADE", "CHECK'D", "APPR'VD", "DESCRIPTION"]
    x_positions = [5668, 5795, 5982, 6162, 6348, 6505]
    for label, xpos in zip(headers, x_positions):
        draw.text((xpos, y1 + 18), label, fill=INK_RGB, font=fonts["tiny"], anchor="la")
    revision_rows_value = scene.metadata.get("revision_rows", [])
    revision_rows = revision_rows_value if isinstance(revision_rows_value, list) else []
    for row_index, row in enumerate(revision_rows, start=1):
        row_y = y1 + 18 + (row_index * row_height)
        values = [row[0], row[1], row[2], row[3], row[2], "ISSUE CONSTR. REV."]
        for value, xpos in zip(values, x_positions):
            draw.text((xpos, row_y), str(value), fill=INK_RGB, font=fonts["tiny"], anchor="la")

    tx1, ty1, tx2, ty2 = TITLE_BLOCK_BOX.to_int_tuple()
    draw.line((tx1, ty1 + 110, tx2, ty1 + 110), fill=INK_RGB, width=1)
    draw.line((tx1, ty1 + 245, tx2, ty1 + 245), fill=INK_RGB, width=1)
    draw.line((tx1, ty1 + 375, tx2, ty1 + 375), fill=INK_RGB, width=1)
    draw.line((tx1, ty1 + 560, tx2, ty1 + 560), fill=INK_RGB, width=1)
    draw.line((tx1 + 385, ty1 + 110, tx1 + 385, ty1 + 375), fill=INK_RGB, width=1)
    draw.line((tx1 + 770, ty1 + 375, tx1 + 770, ty2), fill=INK_RGB, width=1)

    draw.text((6215, ty1 + 70), str(scene.metadata.get("project_name", "SAMPLE Project")), fill=INK_RGB, font=fonts["large"], anchor="mm")
    draw.text((6215, ty1 + 165), str(scene.metadata.get("drawing_title", "SYNTHETIC PROCESS FLOW DIAGRAM")), fill=INK_RGB, font=fonts["medium"], anchor="mm")
    draw.text((6215, ty1 + 205), str(scene.metadata.get("scheme_title", "HYBRID BOOTSTRAP FLOW SCHEME")), fill=INK_RGB, font=fonts["small"], anchor="mm")

    info_rows = [
        ("PROJECT/LOCATION/ASSIGN", scene.metadata.get("project_location", "IPY-65-37"), "ORGANIZATION", scene.metadata.get("organization", "P-0")),
        ("CONTRACTOR PROJECT NO.", scene.metadata.get("contractor_project", "ZX-9409"), "CONTRACTOR NO.", scene.metadata.get("contractor_no", "962")),
        ("DRAWING NAME", scene.metadata.get("sheet_name", "SAMPLE_0000.PNG"), "", ""),
        ("UNIT", scene.metadata.get("unit", "39-6441"), "SCALE", scene.metadata.get("scale", "None")),
        ("DRAW/SHEET NO.", scene.metadata.get("drawing_sheet", "87623869"), "REV", scene.metadata.get("revision", "2")),
    ]
    row_y = ty1 + 270
    for left_label, left_value, right_label, right_value in info_rows:
        draw.text((5650, row_y), left_label, fill=INK_RGB, font=fonts["tiny"], anchor="la")
        draw.text((5935, row_y), str(left_value), fill=INK_RGB, font=fonts["small"], anchor="la")
        if right_label:
            draw.text((6408, row_y), right_label, fill=INK_RGB, font=fonts["tiny"], anchor="la")
            draw.text((6655, row_y), str(right_value), fill=INK_RGB, font=fonts["small"], anchor="la")
        row_y += 92


def _draw_general(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    bbox = node.bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "station_box"))
    text = str(node.metadata.get("text", "STA"))
    if variant == "vessel":
        draw.rounded_rectangle(bbox, radius=max(18, int(node.bbox.width * 0.18)), outline=INK_RGB, width=2)
        caption = str(node.metadata.get("caption", "VESSEL"))
        _multiline(draw, (node.bbox.cx, bbox[3] + 28), caption, fonts["tiny"], anchor="ma")
        _multiline(draw, (node.bbox.cx, bbox[3] + 58), text, fonts["small"], anchor="ma")
    elif variant == "unit_box":
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        _multiline(draw, (node.bbox.cx, node.bbox.cy), text, fonts["small"])
    elif variant == "pump_box":
        x1, y1, x2, y2 = bbox
        body = (x1, y1 + 6, x2 - 14, y2 - 6)
        draw.rounded_rectangle(body, radius=10, outline=INK_RGB, width=2)
        draw.polygon([(x2 - 14, y1 + 10), (x2, (y1 + y2) / 2.0), (x2 - 14, y2 - 10)], outline=INK_RGB)
        _multiline(draw, (node.bbox.cx - 8, node.bbox.cy), text, fonts["tiny"])
    elif variant == "double_box":
        x1, y1, x2, y2 = bbox
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        draw.line((x1, (y1 + y2) / 2.0, x2, (y1 + y2) / 2.0), fill=INK_RGB, width=1)
        _multiline(draw, (node.bbox.cx, node.bbox.cy), text, fonts["tiny"])
    elif variant == "stacked_box":
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        _multiline(draw, (node.bbox.cx, node.bbox.cy), text, fonts["tiny"])
    elif variant == "heater":
        x1, y1, x2, y2 = bbox
        mid_y = (y1 + y2) / 2.0
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        step = max((x2 - x1) / 5.0, 4.0)
        points: list[tuple[float, float]] = [(float(x1) + 4.0, float(mid_y))]
        direction = -1
        current_x = x1 + 4
        while current_x < x2 - 4:
            current_x += step
            points.append((float(current_x), float(mid_y + (direction * (node.bbox.height / 2.6)))))
            direction *= -1
        points.append((float(x2 - 4), float(mid_y)))
        draw.line(points, fill=INK_RGB, width=2)
        draw.text((node.bbox.cx, y1 - 12), text, fill=INK_RGB, font=fonts["tiny"], anchor="ms")
    elif variant == "reducer":
        x1, y1, x2, y2 = bbox
        if node.bbox.width >= node.bbox.height:
            draw.polygon([(x1, y1), (x2, y1 + 8), (x2, y2 - 8), (x1, y2)], outline=INK_RGB)
        else:
            draw.polygon([(x1 + 8, y1), (x2 - 8, y1), (x2, y2), (x1, y2)], outline=INK_RGB)
    elif variant == "orifice":
        x1, y1, x2, y2 = bbox
        draw.line((x1, node.bbox.cy, x2, node.bbox.cy), fill=INK_RGB, width=2)
        draw.ellipse((node.bbox.cx - 10, node.bbox.cy - 10, node.bbox.cx + 10, node.bbox.cy + 10), outline=INK_RGB, width=2)
        draw.text((node.bbox.cx, y1 - 10), text, fill=INK_RGB, font=fonts["tiny"], anchor="ms")
    elif variant == "terminal":
        x1, y1, x2, y2 = bbox
        draw.line((x1, node.bbox.cy, x2, node.bbox.cy), fill=INK_RGB, width=2)
        draw.line((x1 + 4, y1, x1 + 4, y2), fill=INK_RGB, width=2)
        draw.line((x2 - 4, y1, x2 - 4, y2), fill=INK_RGB, width=2)
        draw.text((node.bbox.cx, y1 - 10), text, fill=INK_RGB, font=fonts["tiny"], anchor="ms")
    elif variant == "tag_box":
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        draw.text((node.bbox.cx, node.bbox.cy), text, fill=INK_RGB, font=fonts["tiny"], anchor="mm")
    else:
        draw.rectangle(bbox, outline=INK_RGB, width=2)
        draw.text((node.bbox.cx, node.bbox.cy), text, fill=INK_RGB, font=fonts["small"], anchor="mm")


def _draw_tank(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "vertical_tank"))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=max(14, int(min(node.bbox.width, node.bbox.height) * 0.12)), outline=INK_RGB, width=2)
    if variant == "horizontal_tank":
        draw.line((x1 + 10, node.bbox.cy, x2 - 10, node.bbox.cy), fill=INK_RGB, width=1)
    else:
        draw.line((node.bbox.cx, y1 + 10, node.bbox.cx, y2 - 10), fill=INK_RGB, width=1)
    draw.text((node.bbox.cx, y2 + 16), str(node.metadata.get("caption", "TANK")), fill=INK_RGB, font=fonts["tiny"], anchor="ma")
    draw.text((node.bbox.cx, y2 + 38), str(node.metadata.get("text", "TK-101")), fill=INK_RGB, font=fonts["tiny"], anchor="ma")


def _draw_pump(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    mid_y = (y1 + y2) / 2.0
    draw.ellipse((x1, y1, x2 - 12, y2), outline=INK_RGB, width=2)
    draw.polygon([(x2 - 16, y1 + 10), (x2, mid_y), (x2 - 16, y2 - 10)], outline=INK_RGB)
    draw.line((x1 - 12, mid_y, x1, mid_y), fill=INK_RGB, width=2)
    draw.line((x2, mid_y, x2 + 12, mid_y), fill=INK_RGB, width=2)
    draw.text((node.bbox.cx, y2 + 16), str(node.metadata.get("text", "PU-101")), fill=INK_RGB, font=fonts["tiny"], anchor="ma")


def _draw_inlet_outlet(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "inlet"))
    mid_y = (y1 + y2) / 2.0
    if variant == "terminal_box":
        draw.rectangle((x1, y1, x2, y2), outline=INK_RGB, width=2)
        draw.text((node.bbox.cx, node.bbox.cy), str(node.metadata.get("text", "INLET 1")), fill=INK_RGB, font=fonts["tiny"], anchor="mm")
        return
    draw.line((x1, mid_y, x2 - 18, mid_y), fill=INK_RGB, width=2)
    if variant == "outlet":
        draw.polygon([(x2 - 18, y1 + 4), (x2, mid_y), (x2 - 18, y2 - 4)], outline=INK_RGB, fill=WHITE_RGB)
    else:
        draw.polygon([(x1 + 18, y1 + 4), (x1, mid_y), (x1 + 18, y2 - 4)], outline=INK_RGB, fill=WHITE_RGB)
    draw.text((node.bbox.cx, y1 - 8), str(node.metadata.get("text", "INLET 1")), fill=INK_RGB, font=fonts["tiny"], anchor="ms")


def _draw_valve(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    cx, cy = node.bbox.cx, node.bbox.cy
    draw.polygon([(x1, cy), (cx, y1), (cx, y2)], outline=INK_RGB)
    draw.polygon([(x2, cy), (cx, y1), (cx, y2)], outline=INK_RGB)
    variant = str(node.metadata.get("variant", "gate"))
    if variant == "control":
        draw.line((cx - 10, cy - 10, cx + 10, cy + 10), fill=INK_RGB, width=2)
    elif variant == "check":
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=INK_RGB)
    elif variant == "relief":
        draw.line((cx, y1 - 10, cx, y1 - 24), fill=INK_RGB, width=2)
        draw.polygon([(cx - 8, y1 - 24), (cx + 8, y1 - 24), (cx, y1 - 36)], outline=INK_RGB, fill=WHITE_RGB)
    draw.text((cx, y1 - 12), str(node.metadata.get("tag", "DV-0000")), fill=INK_RGB, font=fonts["tiny"], anchor="ms")


def _draw_instrument(draw: ImageDraw.ImageDraw, node: Node, fonts: dict[str, FontLike]) -> None:
    bbox = node.bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "round"))
    if variant == "panel":
        draw.rectangle(bbox, outline=INK_RGB, width=2)
    else:
        draw.ellipse(bbox, outline=INK_RGB, width=2)
    if variant == "dashed":
        x1, y1, x2, _ = bbox
        _draw_dashed_segment(draw, (x1 + 6, node.bbox.cy), (x2 - 6, node.bbox.cy), dash=8, gap=6, width=1)
    elif variant == "inline":
        x1, _, x2, _ = bbox
        draw.line((x1 + 6, node.bbox.cy, x2 - 6, node.bbox.cy), fill=INK_RGB, width=1)
    _multiline(draw, (node.bbox.cx, node.bbox.cy), str(node.metadata.get("text", "FIT\n101")), fonts["tiny"])


def _draw_connector(draw: ImageDraw.ImageDraw, node: Node) -> None:
    degree_value = node.metadata.get("degree", 0)
    degree = degree_value if isinstance(degree_value, int) else 0
    if degree >= 3:
        draw.ellipse(node.bbox.to_int_tuple(), fill=INK_RGB)


def _draw_crossing(draw: ImageDraw.ImageDraw, node: Node) -> None:
    draw.ellipse(node.bbox.to_int_tuple(), outline=INK_RGB, width=1)


def _draw_arrow(draw: ImageDraw.ImageDraw, scene: Scene, node: Node) -> None:
    cx, cy = node.bbox.cx, node.bbox.cy
    dx, dy = _edge_direction(scene, node)
    angle = atan2(dy, dx)
    size = max(node.bbox.width, node.bbox.height) / 2.0
    tip = (cx + cos(angle) * size, cy + sin(angle) * size)
    left = (cx + cos(angle + (pi * 0.75)) * size * 0.9, cy + sin(angle + (pi * 0.75)) * size * 0.9)
    right = (cx + cos(angle - (pi * 0.75)) * size * 0.9, cy + sin(angle - (pi * 0.75)) * size * 0.9)
    fill = INK_RGB if str(node.metadata.get("variant", "filled")) == "filled" else WHITE_RGB
    draw.polygon([tip, left, right], outline=INK_RGB, fill=fill)


def _draw_edge_labels(draw: ImageDraw.ImageDraw, scene: Scene, fonts: dict[str, FontLike]) -> None:
    for edge in scene.edges:
        text = edge.metadata.get("text")
        if not text:
            continue
        points = _route_edge(scene, edge)
        cx, cy = _polyline_midpoint(points)
        orientation = _polyline_orientation(points)
        if orientation == "horizontal":
            draw.text((cx, cy - 10), str(text), fill=INK_RGB, font=fonts["tiny"], anchor="ms")
        else:
            draw.text((cx + 10, cy), str(text), fill=INK_RGB, font=fonts["tiny"], anchor="ls")


def render_png(scene: Scene, output_path: str | Path) -> None:
    fonts = _fonts()
    image = Image.new("RGB", (scene.width, scene.height), PAPER_RGB)
    draw = ImageDraw.Draw(image)

    _draw_sheet_template(draw, scene, fonts)
    for edge in scene.edges:
        points = _route_edge(scene, edge)
        _draw_polyline(draw, points, dashed=edge.style == "non-solid", width=2)

    for node in scene.sorted_nodes():
        if node.label == "background":
            continue
        if node.label == "general":
            _draw_general(draw, node, fonts)
        elif node.label == "tank":
            _draw_tank(draw, node, fonts)
        elif node.label == "pump":
            _draw_pump(draw, node, fonts)
        elif node.label == "inlet/outlet":
            _draw_inlet_outlet(draw, node, fonts)
        elif node.label == "valve":
            _draw_valve(draw, node, fonts)
        elif node.label == "instrumentation":
            _draw_instrument(draw, node, fonts)
        elif node.label == "connector":
            _draw_connector(draw, node)
        elif node.label == "crossing":
            _draw_crossing(draw, node)
        elif node.label == "arrow":
            _draw_arrow(draw, scene, node)

    _draw_edge_labels(draw, scene, fonts)
    image.save(Path(output_path))


def _svg_rect(bbox: BoundingBox, stroke: str = SVG_INK, fill: str = "none", width: int = 2, dash: str | None = None, rx: int = 0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{bbox.xmin:.1f}" y="{bbox.ymin:.1f}" width="{bbox.width:.1f}" height="{bbox.height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" rx="{rx}" ry="{rx}"{dash_attr} />'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, width: int = 2) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{SVG_INK}" stroke-width="{width}" />'


def _svg_polyline(points: list[tuple[float, float]], dashed: bool = False, width: int = 2) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = ' stroke-dasharray="20 12"' if dashed else ""
    return f'<polyline points="{coords}" fill="none" stroke="{SVG_INK}" stroke-width="{width}"{dash_attr} />'


def _svg_text(x: float, y: float, text: str, size: int = 20, anchor: str = "middle", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    lines = [escape(line) for line in str(text).split("\n")]
    if len(lines) == 1:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{SVG_INK}"{transform}>{lines[0]}</text>'
        )
    tspans = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else "1.1em"
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{line}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{SVG_INK}"{transform}>{"".join(tspans)}</text>'
    )


def _svg_sheet_template(scene: Scene) -> list[str]:
    parts = [
        f'<rect x="0" y="0" width="{scene.width}" height="{scene.height}" fill="{SVG_PAPER}" />',
        _svg_rect(FRAME_BOX, dash="18 10"),
        _svg_rect(NOTES_BOX),
        _svg_rect(REVISION_BOX),
        _svg_rect(TITLE_BLOCK_BOX),
        _svg_text(6215, 258, "NOTES", size=26),
    ]
    y = 310
    notes_value = scene.metadata.get("notes", [])
    notes = notes_value if isinstance(notes_value, list) else []
    for index, note in enumerate(notes, start=1):
        parts.append(_svg_text(5660, y, f"{index}.", size=16, anchor="start"))
        parts.append(_svg_text(5715, y, str(note), size=16, anchor="start"))
        y += 66

    x1, y1, x2, y2 = REVISION_BOX.to_int_tuple()
    row_height = 58
    for offset in range(1, 5):
        parts.append(_svg_line(x1, y1 + offset * row_height, x2, y1 + offset * row_height, width=1))
    for xpos in (5735, 5925, 6115, 6285, 6465):
        parts.append(_svg_line(xpos, y1, xpos, y1 + (row_height * 5), width=1))
    headers = ["ISSUE", "DATE", "MADE", "CHECK'D", "APPR'VD", "DESCRIPTION"]
    x_positions = [5668, 5795, 5982, 6162, 6348, 6505]
    for label, xpos in zip(headers, x_positions):
        parts.append(_svg_text(xpos, y1 + 18, label, size=16, anchor="start"))
    revision_rows_value = scene.metadata.get("revision_rows", [])
    revision_rows = revision_rows_value if isinstance(revision_rows_value, list) else []
    for row_index, row in enumerate(revision_rows, start=1):
        row_y = y1 + 18 + (row_index * row_height)
        values = [row[0], row[1], row[2], row[3], row[2], "ISSUE CONSTR. REV."]
        for value, xpos in zip(values, x_positions):
            parts.append(_svg_text(xpos, row_y, str(value), size=16, anchor="start"))

    tx1, ty1, tx2, ty2 = TITLE_BLOCK_BOX.to_int_tuple()
    for line_y in (ty1 + 110, ty1 + 245, ty1 + 375, ty1 + 560):
        parts.append(_svg_line(tx1, line_y, tx2, line_y, width=1))
    parts.append(_svg_line(tx1 + 385, ty1 + 110, tx1 + 385, ty1 + 375, width=1))
    parts.append(_svg_line(tx1 + 770, ty1 + 375, tx1 + 770, ty2, width=1))
    parts.extend(
        [
            _svg_text(6215, ty1 + 70, str(scene.metadata.get("project_name", "SAMPLE Project")), size=34),
            _svg_text(6215, ty1 + 165, str(scene.metadata.get("drawing_title", "SYNTHETIC PROCESS FLOW DIAGRAM")), size=26),
            _svg_text(6215, ty1 + 205, str(scene.metadata.get("scheme_title", "HYBRID BOOTSTRAP FLOW SCHEME")), size=20),
        ]
    )
    info_rows = [
        ("PROJECT/LOCATION/ASSIGN", scene.metadata.get("project_location", "IPY-65-37"), "ORGANIZATION", scene.metadata.get("organization", "P-0")),
        ("CONTRACTOR PROJECT NO.", scene.metadata.get("contractor_project", "ZX-9409"), "CONTRACTOR NO.", scene.metadata.get("contractor_no", "962")),
        ("DRAWING NAME", scene.metadata.get("sheet_name", "SAMPLE_0000.PNG"), "", ""),
        ("UNIT", scene.metadata.get("unit", "39-6441"), "SCALE", scene.metadata.get("scale", "None")),
        ("DRAW/SHEET NO.", scene.metadata.get("drawing_sheet", "87623869"), "REV", scene.metadata.get("revision", "2")),
    ]
    row_y = ty1 + 270
    for left_label, left_value, right_label, right_value in info_rows:
        parts.append(_svg_text(5650, row_y, left_label, size=16, anchor="start"))
        parts.append(_svg_text(5935, row_y, str(left_value), size=20, anchor="start"))
        if right_label:
            parts.append(_svg_text(6408, row_y, right_label, size=16, anchor="start"))
            parts.append(_svg_text(6655, row_y, str(right_value), size=20, anchor="start"))
        row_y += 92
    return parts


def _svg_general(node: Node) -> list[str]:
    bbox = node.bbox
    variant = str(node.metadata.get("variant", "station_box"))
    text = str(node.metadata.get("text", "STA"))
    x1, y1, x2, y2 = bbox.to_int_tuple()
    parts: list[str] = []
    if variant == "vessel":
        parts.append(_svg_rect(bbox, rx=max(18, int(bbox.width * 0.18))))
        parts.append(_svg_text(bbox.cx, y2 + 28, str(node.metadata.get("caption", "VESSEL")), size=16))
        parts.append(_svg_text(bbox.cx, y2 + 58, text, size=20))
    elif variant == "unit_box":
        parts.append(_svg_rect(bbox))
        parts.append(_svg_text(bbox.cx, bbox.cy, text, size=20))
    elif variant == "pump_box":
        body = BoundingBox(x1, y1 + 6, x2 - 14, y2 - 6)
        parts.append(_svg_rect(body, rx=10))
        point_text = f"{x2 - 14:.1f},{y1 + 10:.1f} {x2:.1f},{(y1 + y2) / 2.0:.1f} {x2 - 14:.1f},{y2 - 10:.1f}"
        parts.append(f'<polygon points="{point_text}" fill="none" stroke="{SVG_INK}" stroke-width="2" />')
        parts.append(_svg_text(bbox.cx - 8, bbox.cy, text, size=16))
    elif variant == "double_box":
        parts.append(_svg_rect(bbox))
        parts.append(_svg_line(x1, (y1 + y2) / 2.0, x2, (y1 + y2) / 2.0, width=1))
        parts.append(_svg_text(bbox.cx, bbox.cy, text, size=16))
    elif variant == "stacked_box":
        parts.append(_svg_rect(bbox))
        parts.append(_svg_text(bbox.cx, bbox.cy, text, size=16))
    elif variant == "heater":
        mid_y = (y1 + y2) / 2.0
        step = max((x2 - x1) / 5.0, 4.0)
        points: list[tuple[float, float]] = [(float(x1) + 4.0, float(mid_y))]
        direction = -1
        current_x = x1 + 4
        while current_x < x2 - 4:
            current_x += step
            points.append((float(current_x), float(mid_y + (direction * (bbox.height / 2.6)))))
            direction *= -1
        points.append((float(x2 - 4), float(mid_y)))
        parts.append(_svg_rect(bbox))
        parts.append(_svg_polyline(points))
        parts.append(_svg_text(bbox.cx, y1 - 12, text, size=16))
    elif variant == "reducer":
        if bbox.width >= bbox.height:
            coords = [(x1, y1), (x2, y1 + 8), (x2, y2 - 8), (x1, y2)]
        else:
            coords = [(x1 + 8, y1), (x2 - 8, y1), (x2, y2), (x1, y2)]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polygon points="{point_text}" fill="none" stroke="{SVG_INK}" stroke-width="2" />')
    elif variant == "orifice":
        parts.append(_svg_line(x1, bbox.cy, x2, bbox.cy))
        parts.append(f'<circle cx="{bbox.cx:.1f}" cy="{bbox.cy:.1f}" r="10" fill="none" stroke="{SVG_INK}" stroke-width="2" />')
        parts.append(_svg_text(bbox.cx, y1 - 10, text, size=16))
    elif variant == "terminal":
        parts.append(_svg_line(x1, bbox.cy, x2, bbox.cy))
        parts.append(_svg_line(x1 + 4, y1, x1 + 4, y2))
        parts.append(_svg_line(x2 - 4, y1, x2 - 4, y2))
        parts.append(_svg_text(bbox.cx, y1 - 10, text, size=16))
    else:
        parts.append(_svg_rect(bbox))
        parts.append(_svg_text(bbox.cx, bbox.cy, text, size=16 if variant == "tag_box" else 20))
    return parts


def _svg_tank(node: Node) -> list[str]:
    bbox = node.bbox
    x1, y1, x2, y2 = bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "vertical_tank"))
    parts = [_svg_rect(bbox, rx=max(14, int(min(bbox.width, bbox.height) * 0.12)))]
    if variant == "horizontal_tank":
        parts.append(_svg_line(x1 + 10, bbox.cy, x2 - 10, bbox.cy, width=1))
    else:
        parts.append(_svg_line(bbox.cx, y1 + 10, bbox.cx, y2 - 10, width=1))
    parts.append(_svg_text(bbox.cx, y2 + 16, str(node.metadata.get("caption", "TANK")), size=16))
    parts.append(_svg_text(bbox.cx, y2 + 38, str(node.metadata.get("text", "TK-101")), size=16))
    return parts


def _svg_pump(node: Node) -> list[str]:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    mid_y = (y1 + y2) / 2.0
    parts = [
        f'<ellipse cx="{(x1 + x2 - 12) / 2.0:.1f}" cy="{mid_y:.1f}" rx="{(x2 - x1 - 12) / 2.0:.1f}" ry="{(y2 - y1) / 2.0:.1f}" fill="none" stroke="{SVG_INK}" stroke-width="2" />',
        f'<polygon points="{x2 - 16:.1f},{y1 + 10:.1f} {x2:.1f},{mid_y:.1f} {x2 - 16:.1f},{y2 - 10:.1f}" fill="none" stroke="{SVG_INK}" stroke-width="2" />',
        _svg_line(x1 - 12, mid_y, x1, mid_y),
        _svg_line(x2, mid_y, x2 + 12, mid_y),
        _svg_text(node.bbox.cx, y2 + 16, str(node.metadata.get("text", "PU-101")), size=16),
    ]
    return parts


def _svg_inlet_outlet(node: Node) -> list[str]:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    variant = str(node.metadata.get("variant", "inlet"))
    mid_y = (y1 + y2) / 2.0
    if variant == "terminal_box":
        return [_svg_rect(node.bbox), _svg_text(node.bbox.cx, node.bbox.cy, str(node.metadata.get("text", "INLET 1")), size=16)]
    if variant == "outlet":
        polygon = f'{x2 - 18:.1f},{y1 + 4:.1f} {x2:.1f},{mid_y:.1f} {x2 - 18:.1f},{y2 - 4:.1f}'
    else:
        polygon = f'{x1 + 18:.1f},{y1 + 4:.1f} {x1:.1f},{mid_y:.1f} {x1 + 18:.1f},{y2 - 4:.1f}'
    return [
        _svg_line(x1, mid_y, x2 - 18, mid_y),
        f'<polygon points="{polygon}" fill="{SVG_PAPER}" stroke="{SVG_INK}" stroke-width="2" />',
        _svg_text(node.bbox.cx, y1 - 8, str(node.metadata.get("text", "INLET 1")), size=16),
    ]


def _svg_valve(node: Node) -> list[str]:
    x1, y1, x2, y2 = node.bbox.to_int_tuple()
    cx, cy = node.bbox.cx, node.bbox.cy
    left = f"{x1:.1f},{cy:.1f} {cx:.1f},{y1:.1f} {cx:.1f},{y2:.1f}"
    right = f"{x2:.1f},{cy:.1f} {cx:.1f},{y1:.1f} {cx:.1f},{y2:.1f}"
    parts = [
        f'<polygon points="{left}" fill="none" stroke="{SVG_INK}" stroke-width="2" />',
        f'<polygon points="{right}" fill="none" stroke="{SVG_INK}" stroke-width="2" />',
        _svg_text(cx, y1 - 12, str(node.metadata.get("tag", "DV-0000")), size=16),
    ]
    variant = str(node.metadata.get("variant", "gate"))
    if variant == "control":
        parts.append(_svg_line(cx - 10, cy - 10, cx + 10, cy + 10))
    elif variant == "check":
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{SVG_INK}" />')
    elif variant == "relief":
        parts.append(_svg_line(cx, y1 - 10, cx, y1 - 24))
        parts.append(f'<polygon points="{cx - 8:.1f},{y1 - 24:.1f} {cx + 8:.1f},{y1 - 24:.1f} {cx:.1f},{y1 - 36:.1f}" fill="{SVG_PAPER}" stroke="{SVG_INK}" stroke-width="2" />')
    return parts


def _svg_instrument(node: Node) -> list[str]:
    radius = max(node.bbox.width, node.bbox.height) / 2.0
    variant = str(node.metadata.get("variant", "round"))
    parts: list[str] = []
    if variant == "panel":
        parts.append(_svg_rect(node.bbox))
    else:
        parts.append(f'<circle cx="{node.bbox.cx:.1f}" cy="{node.bbox.cy:.1f}" r="{radius:.1f}" fill="none" stroke="{SVG_INK}" stroke-width="2" />')
    if variant == "dashed":
        parts.append(f'<line x1="{node.bbox.xmin + 6:.1f}" y1="{node.bbox.cy:.1f}" x2="{node.bbox.xmax - 6:.1f}" y2="{node.bbox.cy:.1f}" stroke="{SVG_INK}" stroke-width="1" stroke-dasharray="8 6" />')
    elif variant == "inline":
        parts.append(_svg_line(node.bbox.xmin + 6, node.bbox.cy, node.bbox.xmax - 6, node.bbox.cy, width=1))
    parts.append(_svg_text(node.bbox.cx, node.bbox.cy, str(node.metadata.get("text", "FIT\n101")), size=16))
    return parts


def _svg_connector(node: Node) -> list[str]:
    degree_value = node.metadata.get("degree", 0)
    degree = degree_value if isinstance(degree_value, int) else 0
    if degree < 3:
        return []
    radius = max(node.bbox.width, node.bbox.height) / 2.0
    return [f'<circle cx="{node.bbox.cx:.1f}" cy="{node.bbox.cy:.1f}" r="{radius:.1f}" fill="{SVG_INK}" />']


def _svg_crossing(node: Node) -> list[str]:
    radius = max(node.bbox.width, node.bbox.height) / 2.0
    return [f'<circle cx="{node.bbox.cx:.1f}" cy="{node.bbox.cy:.1f}" r="{radius:.1f}" fill="none" stroke="{SVG_INK}" stroke-width="1" />']


def _svg_arrow(scene: Scene, node: Node) -> list[str]:
    cx, cy = node.bbox.cx, node.bbox.cy
    dx, dy = _edge_direction(scene, node)
    angle = atan2(dy, dx)
    size = max(node.bbox.width, node.bbox.height) / 2.0
    tip = (cx + cos(angle) * size, cy + sin(angle) * size)
    left = (cx + cos(angle + (pi * 0.75)) * size * 0.9, cy + sin(angle + (pi * 0.75)) * size * 0.9)
    right = (cx + cos(angle - (pi * 0.75)) * size * 0.9, cy + sin(angle - (pi * 0.75)) * size * 0.9)
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in [tip, left, right])
    fill = SVG_INK if str(node.metadata.get("variant", "filled")) == "filled" else SVG_PAPER
    return [f'<polygon points="{point_text}" fill="{fill}" stroke="{SVG_INK}" stroke-width="2" />']


def render_svg(scene: Scene, output_path: str | Path) -> None:
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" '
            f'viewBox="0 0 {scene.width} {scene.height}">'
        ),
    ]
    parts.extend(_svg_sheet_template(scene))
    for edge in scene.edges:
        parts.append(_svg_polyline(_route_edge(scene, edge), dashed=edge.style == "non-solid"))
    for node in scene.sorted_nodes():
        if node.label == "background":
            continue
        if node.label == "general":
            parts.extend(_svg_general(node))
        elif node.label == "tank":
            parts.extend(_svg_tank(node))
        elif node.label == "pump":
            parts.extend(_svg_pump(node))
        elif node.label == "inlet/outlet":
            parts.extend(_svg_inlet_outlet(node))
        elif node.label == "valve":
            parts.extend(_svg_valve(node))
        elif node.label == "instrumentation":
            parts.extend(_svg_instrument(node))
        elif node.label == "connector":
            parts.extend(_svg_connector(node))
        elif node.label == "crossing":
            parts.extend(_svg_crossing(node))
        elif node.label == "arrow":
            parts.extend(_svg_arrow(scene, node))
    for edge in scene.edges:
        text = edge.metadata.get("text")
        if not text:
            continue
        points = _route_edge(scene, edge)
        cx, cy = _polyline_midpoint(points)
        rotate = -90 if _polyline_orientation(points) == "vertical" else None
        parts.append(_svg_text(cx, cy - 8, str(text), size=16, rotate=rotate))
    parts.append('</svg>')
    Path(output_path).write_text("\n".join(parts), encoding="utf-8")
