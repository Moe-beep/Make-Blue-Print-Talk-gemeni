"""Room and apartment net areas from a vector floor-plan PDF (Swiss Dwellings style), no model needed.

The PDF carries the exact geometry: every room is a filled vector polygon, every apartment a
stroke-only outline, and the drawing scale ("1:100") is printed in the title block. So:
  room area      = area of the smallest filled region (outline minus holes) in the room style
                   (pdf stroke shared by most rooms) containing the room number, in pt^2,
                   converted with the printed scale (1 pt = 25.4/72 mm on paper)
  room label     = the text line just above the room number ("Stanza" over "5.03")
  apartment      = the smallest stroke-only outline containing its marker ("App. A", "Whg. B");
                   its rooms are those whose number lies inside that outline
Net sums leave out outdoor rooms (balcony, loggia, terrace labels) and shafts/voids (unlabeled rooms
in a fill colour no labeled room uses). Sums use the unrounded room areas; values are rounded only
when written out.
"""

import ctypes
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from pypdfium2 import raw

PDF_PLAN = "pdf_plan"  # source name in the facts file
PT_TO_M = 0.0254 / 72
AREA_DECIMALS = 2
PRECISE_DECIMALS = 4
STYLE_WIDTH_DECIMALS = 2
SCALE_RE = re.compile(r"\b1\s?:\s?(\d{2,4})\b")
ROOM_NUMBER_RE = re.compile(r"[A-Z]?\d+\.\d{2}")
APARTMENT_RE = re.compile(r"(?:Whg|App|Appt|Apt|Unit|Wohnung)\.?\s*([A-Z0-9]{1,3})")
# label line above a room number: horizontal / vertical offset of the line centres, in pt
LABEL_MAX_DX = 15.0
LABEL_DY_RANGE = (3.0, 12.0)
# labels of outdoor spaces, excluded from net floor areas (schema: balconies, loggias, terraces)
OUTDOOR_WORDS = (
    "balkon",
    "balcon",
    "balcone",
    "loggia",
    "terrasse",
    "terrazza",
    "terrace",
    "veranda",
    "parveke",
)


@dataclass(frozen=True)
class TextLine:
    text: str
    x: float
    y: float


def text_lines(textpage: pdfium.PdfTextPage) -> list[TextLine]:
    lines = []
    for i in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(i)
        text = textpage.get_text_bounded(left, bottom, right, top).strip()
        if text:
            lines.append(TextLine(text, (left + right) / 2, (bottom + top) / 2))
    return lines


def path_subpaths(obj: pdfium.PdfObject) -> list[np.ndarray]:
    """Closed pieces of a path object (split at each move-to), in page coordinates."""
    matrix = raw.FS_MATRIX()
    raw.FPDFPageObj_GetMatrix(obj.raw, matrix)
    subpaths, current = [], []
    for i in range(raw.FPDFPath_CountSegments(obj.raw)):
        segment = raw.FPDFPath_GetPathSegment(obj.raw, i)
        x, y = ctypes.c_float(), ctypes.c_float()
        raw.FPDFPathSegment_GetPoint(segment, x, y)
        if raw.FPDFPathSegment_GetType(segment) == raw.FPDF_SEGMENT_MOVETO and current:
            subpaths.append(np.asarray(current))
            current = []
        current.append(
            (
                matrix.a * x.value + matrix.c * y.value + matrix.e,
                matrix.b * x.value + matrix.d * y.value + matrix.f,
            )
        )
    if current:
        subpaths.append(np.asarray(current))
    return [sub for sub in subpaths if len(sub) >= 3]


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def contains(poly: np.ndarray, x: float, y: float) -> bool:
    """Even-odd point-in-polygon test."""
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


@dataclass(frozen=True, eq=False)  # holds numpy arrays: compare by identity
class Region:
    """One outer boundary with the holes cut out of it (columns, shafts inside a room)."""

    outer: np.ndarray
    holes: tuple[np.ndarray, ...]
    style: tuple = ()  # (stroke colour, stroke width) of the path it came from
    fill: tuple | None = None  # fill colour (r, g, b) of the path, None for stroke-only paths

    @property
    def area(self) -> float:
        return polygon_area(self.outer) - sum(polygon_area(h) for h in self.holes)

    def contains(self, x: float, y: float) -> bool:
        return contains(self.outer, x, y) and not any(contains(h, x, y) for h in self.holes)


def regions_of(
    subpaths: list[np.ndarray], style: tuple = (), fill: tuple | None = None
) -> list[Region]:
    """Group the pieces of one path: a piece inside a larger piece is a hole of it.

    A path with several separate pieces (e.g. all walls in one path) gives several regions, so its
    combined outline can never be mistaken for a room.
    """
    by_size = sorted(subpaths, key=polygon_area, reverse=True)
    outers: list[np.ndarray] = []
    holes: dict[int, list[np.ndarray]] = {}
    for piece in by_size:
        x, y = piece[0]
        parent = next((i for i in reversed(range(len(outers))) if contains(outers[i], x, y)), None)
        if parent is None:
            outers.append(piece)
            holes[len(outers) - 1] = []
        else:
            holes[parent].append(piece)
    return [Region(outer, tuple(holes[i]), style, fill) for i, outer in enumerate(outers)]


def page_regions(page: pdfium.PdfPage) -> tuple[list[Region], list[Region]]:
    """(filled regions = rooms and other fills, stroke-only regions = outlines)."""
    filled, outlines = [], []
    for obj in page.get_objects(max_depth=1):
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        fill_mode, stroke = ctypes.c_int(), ctypes.c_int()
        raw.FPDFPath_GetDrawMode(obj.raw, fill_mode, stroke)
        fill = path_fill(obj) if fill_mode.value else None
        regions = regions_of(path_subpaths(obj), path_style(obj, bool(stroke.value)), fill)
        (filled if fill_mode.value else outlines).extend(regions)
    return filled, outlines


def path_fill(obj: pdfium.PdfObject) -> tuple[int, int, int]:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    raw.FPDFPageObj_GetFillColor(obj.raw, r, g, b, a)
    return (r.value, g.value, b.value)


def path_style(obj: pdfium.PdfObject, stroked: bool) -> tuple:
    """Stroke colour and width: rooms, fixtures and walls are drawn in different strokes."""
    if not stroked:
        return (None, None)
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    raw.FPDFPageObj_GetStrokeColor(obj.raw, r, g, b, a)
    width = ctypes.c_float()
    raw.FPDFPageObj_GetStrokeWidth(obj.raw, width)
    return ((r.value, g.value, b.value), round(width.value, STYLE_WIDTH_DECIMALS))


def room_regions(filled: list[Region], numbers: list[TextLine]) -> list[Region]:
    """Filled regions drawn in the room style.

    The room style is the most common style of the smallest region around each room number:
    a fixture symbol (bathtub, kitchen unit) under a number is rarer than the room itself.
    """
    styles = Counter(
        region.style
        for number in numbers
        if (region := smallest_containing(filled, number.x, number.y)) is not None
    )
    if not styles:
        return filled
    room_style = styles.most_common(1)[0][0]
    return [region for region in filled if region.style == room_style]


def smallest_containing(regions: list[Region], x: float, y: float) -> Region | None:
    containing = [r for r in regions if r.contains(x, y)]
    return min(containing, key=lambda r: r.area) if containing else None


def label_above(number: TextLine, lines: list[TextLine]) -> str | None:
    low, high = LABEL_DY_RANGE
    candidates = [
        line
        for line in lines
        if abs(line.x - number.x) < LABEL_MAX_DX
        and low < line.y - number.y < high
        and not re.search(r"\d", line.text)
        and not APARTMENT_RE.fullmatch(line.text)
    ]
    return (
        min(candidates, key=lambda line: line.y - number.y).text if candidates else None
    )


def is_outdoor(label: str | None) -> bool:
    return label is not None and any(word in label.lower() for word in OUTDOOR_WORDS)


def extract_pdf_plan(pdf_path: Path, page_index: int = 0) -> dict:
    """Rooms (number -> label, area, apartment) and apartment / floor net areas of one plan page."""
    page = pdfium.PdfDocument(str(pdf_path))[page_index]
    lines = text_lines(page.get_textpage())
    scale_match = next(
        (match for line in lines if (match := SCALE_RE.search(line.text))), None
    )
    if scale_match is None:
        raise ValueError(
            f"{pdf_path}: no drawing scale (e.g. '1:100') found in the text layer"
        )
    scale = int(scale_match.group(1))
    m2_per_pt2 = (PT_TO_M * scale) ** 2
    filled, outlines = page_regions(page)
    numbers = [line for line in lines if ROOM_NUMBER_RE.fullmatch(line.text)]
    rooms_filled = room_regions(filled, numbers)

    apartments = {}
    for line in lines:
        match = APARTMENT_RE.fullmatch(line.text)
        outline = smallest_containing(outlines, line.x, line.y) if match else None
        if outline is not None:
            apartments[match.group(1)] = outline

    rooms = {}
    for line in numbers:
        region = smallest_containing(rooms_filled, line.x, line.y)
        if region is None:
            continue
        apartment = next(
            (
                k
                for k, outline in apartments.items()
                if outline.contains(line.x, line.y)
            ),
            None,
        )
        rooms[line.text] = {
            "label": label_above(line, lines),
            "area_exact": region.area * m2_per_pt2,
            "polygon_pt": [[round(float(x), 2), round(float(y), 2)] for x, y in region.outer],
            "area_pt2": round(region.area, 1),
            "apartment": apartment,
            "fill": region.fill,
        }

    # an unlabeled room in a fill colour that no labeled room uses is a shaft or void (grey hatching)
    labeled_fills = {r["fill"] for r in rooms.values() if r["label"]}
    for r in rooms.values():
        if is_outdoor(r["label"]):
            r["excluded"] = "outdoor"
        elif r["label"] is None and labeled_fills and r["fill"] not in labeled_fills:
            r["excluded"] = "shaft_or_void"
        else:
            r["excluded"] = None

    def net_sum(selected: list[dict]) -> float:
        return round(sum(r["area_exact"] for r in selected if r["excluded"] is None), AREA_DECIMALS)

    return {
        "pdf": str(pdf_path),
        "scale_denominator": scale,
        "rooms": {
            number: {
                "label": r["label"],
                "net_floor_area_m2": round(r["area_exact"], AREA_DECIMALS),
                # unrounded enough for sums (building graph); net_floor_area_m2 is the value to show
                "area_m2_precise": round(r["area_exact"], PRECISE_DECIMALS),
                "apartment": r["apartment"],
                "polygon_pt": r["polygon_pt"],  # outer outline, PDF points (y up)
                "area_pt2": r["area_pt2"],  # outline minus holes
                "excluded_from_net_area": r["excluded"],
            }
            for number, r in rooms.items()
        },
        "apartments": {
            key: {
                "net_floor_area_m2": net_sum(
                    [r for r in rooms.values() if r["apartment"] == key]
                )
            }
            for key in sorted(apartments)
        },
        "total_net_floor_area_m2": net_sum(list(rooms.values())),
    }
