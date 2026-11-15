"""Plan graph = layout rooms + OCR texts, linked by position. Pure code, no model calls."""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter

import cv2
import numpy as np

from .ocr import OcrWord
from .schemas import GraphObject, GraphRoom, GraphText, GraphWall, PlanGraph, TextKind

AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(m\s*[2²]|m\^2|qm|sqm|sq\.?\s*m|sq\.?\s*ft|ft\s*[2²]|sf)(?![a-z])", re.I)
LABEL_AREA_RE = re.compile(r"^([^\W\d_][\w.\-/]{0,15})\s+(\d+(?:[.,]\d+)?\s*\S{1,4})$")  # 'Kind 10,86 m²'
NUMBER_RE = re.compile(r"^\d{1,6}(?:[.,]\d{1,3})?$")
DECIMAL_RE = re.compile(r"^\d{1,3}[.,]\d{1,2}$")   # '5,8' without unit
LABEL_RE = re.compile(r"^[^\W\d_]{1,8}\d{0,2}$")    # MH, KPH, WC, MH2, Küche
ROOM_NUMBER_RE = re.compile(r"^([A-Z]?\d{1,2})\.(\d{2})$")   # '1.04', 'A2.13' (dot only: '5,80' is an area)
APARTMENT_RE = re.compile(r"^(?:(?:Whg|App|Appt|Apt|Apartment|Unit|Flat|Wohnung|As|Asunto|Lgh|Leil)(?:\.\s*|\s+)"
                          r"([A-Z0-9]{1,3})|([A-D]))$", re.I)
SINGLE_LETTER_RE = re.compile(r"^[A-D]$")   # bare letter, upper case only; E.. excluded: H = huone in Finnish
MIN_ROOM_NUMBERS = 2          # distinct numbers sharing a prefix ('1.01', '1.02') before they count as room numbers
MAX_ROOM_SUFFIX = 60          # room numbers run 01..~40 per storey; larger suffixes look like areas
SCALE_MIN_SAMPLES = {"room_areas": 2, "dimensions": 3}
SCALE_MAX_SPREAD = 0.15       # median relative deviation allowed between the samples of one scale
DIM_MM_MIN = 100              # integer dimensions >= 100 are millimetres ('3600'), decimals < 50 metres ('4,20')
DIM_M_MAX = 50


def parse_area(text: str) -> tuple[float, str] | None:
    m = AREA_RE.search(text.replace(" ", " "))
    if not m:
        return None
    value, unit = float(m.group(1).replace(",", ".")), m.group(2).lower().replace(" ", "")
    return (value * 0.09290304, "sqft") if ("ft" in unit or unit == "sf") else (value, "m2")


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# separate thresholds: short OCR labels ('WC', 'S', 'MH') and small symbols are often read
# correctly with low confidence, a wrong room class is more harmful
DEFAULT_MIN_CONF = {"texts": 0.6, "rooms": 0.8, "objects": 0.6}


def thresholds(min_conf: float | dict | None = None) -> dict[str, float]:
    """One number = same threshold for texts, rooms and objects; dict = per kind (missing -> default)."""
    if min_conf is None:
        return dict(DEFAULT_MIN_CONF)
    if isinstance(min_conf, (int, float)):
        return {k: float(min_conf) for k in DEFAULT_MIN_CONF}
    return {**DEFAULT_MIN_CONF, **{k: float(v) for k, v in min_conf.items() if v is not None}}


def build_graph(source: str, page: int, size: tuple[int, int], words: list[OcrWord],
                rooms: list[GraphRoom] | None = None, min_conf: float | dict | None = None,
                objects: list[GraphObject] | None = None, floors: list[list[int]] | None = None,
                walls: list[GraphWall] | None = None) -> PlanGraph:
    """Detections below their threshold (OCR text, room class, symbol) are dropped here, so
    uncertain evidence never reaches the answers. Their number is kept in graph.dropped.
    Mutates rooms/objects (ids, labels): pass copies when building several graphs from one detection set."""
    th = thresholds(min_conf)
    rooms, objects, walls = rooms or [], objects or [], walls or []
    dropped = {"texts": sum(w.conf < th["texts"] for w in words), "rooms": sum(r.type_conf < th["rooms"] for r in rooms),
               "objects": sum(o.conf < th["objects"] for o in objects)}
    rejected_rooms = [r for r in rooms if r.type_conf < th["rooms"]]
    rejected_objects = [o for o in objects if o.conf < th["objects"]]
    for i, r in enumerate(rejected_rooms, 1):
        r.id = f"rx{i:02d}"
    for i, o in enumerate(rejected_objects, 1):
        o.id = f"ox{i:03d}"
    words = [w for w in words if w.conf >= th["texts"]]
    rooms = [r for r in rooms if r.type_conf >= th["rooms"]]
    objects = [o for o in objects if o.conf >= th["objects"]]
    for i, r in enumerate(rooms, 1):  # keep IDs consecutive after filtering
        r.id = f"r{i:02d}"
    for i, o in enumerate(objects, 1):
        o.id = f"o{i:03d}"
    texts = [_classify(GraphText(id=f"_{i}", text=w.text, box_px=[round(v) for v in w.box], conf=w.conf), th["texts"])
             for i, w in enumerate(_split_label_area(words))]
    _mark_room_numbers(texts)
    _link(texts, rooms)
    _finalize(rooms, texts)
    polys = [np.array(r.polygon_px, np.int32).reshape(-1, 1, 2) for r in rooms]
    for o in objects:  # symbol -> room it stands in (doors/windows sit on walls: tolerance)
        room = _room_at(rooms, polys, _center(o.box_px), tol=max(6, o.box_px[3] - o.box_px[1]))
        o.room_id = room.id if room else None
    for o in rejected_objects:
        room = _room_at(rooms, polys, _center(o.box_px), tol=max(6, o.box_px[3] - o.box_px[1]))
        o.room_id = room.id if room else None

    texts.sort(key=lambda t: (t.box_px[1], t.box_px[0]))  # stable IDs in reading order
    ids = {t.id: f"t{i:03d}" for i, t in enumerate(texts, 1)}
    for t in texts:
        t.id = ids[t.id]
    for r in rooms:
        r.label_text_ids = [ids[i] for i in r.label_text_ids]
        r.area_text_ids = [ids[i] for i in r.area_text_ids]
    return PlanGraph(source=source, page=page, image_size=list(size), rooms=rooms, texts=texts,
                     objects=objects, walls=walls, rejected_rooms=rejected_rooms, rejected_objects=rejected_objects,
                     scale=estimate_scale(rooms, texts, walls), floors=floors or [], min_conf=th, dropped=dropped)


def _split_areas(w: OcrWord) -> list[OcrWord] | None:
    """'3.94 m² 4.04 m²' (two neighbouring areas joined into one line) -> one word per area."""
    hits = list(AREA_RE.finditer(w.text))
    if len(hits) < 2 or w.vertical or AREA_RE.sub("", w.text).strip():
        return None
    width, n = w.box[2] - w.box[0], len(w.text)
    return [OcrWord(m.group(0), [w.box[0] + width * m.start() / n, w.box[1], w.box[0] + width * m.end() / n, w.box[3]],
                    w.conf) for m in hits]


def _split_label_area(words: list[OcrWord]) -> list[OcrWord]:
    """'Kind 10,86 m²' on one OCR line -> label + area, box split by character count; two areas -> two words."""
    out = []
    for w in words:
        if parts := _split_areas(w):
            out += parts
            continue
        m = LABEL_AREA_RE.match(w.text.strip())
        if m and not w.vertical and parse_area(m.group(2)):
            cut = w.box[0] + (w.box[2] - w.box[0]) * (len(m.group(1)) + 0.5) / len(w.text)
            out += [OcrWord(m.group(1), [w.box[0], w.box[1], cut, w.box[3]], w.conf),
                    OcrWord(m.group(2), [cut, w.box[1], w.box[2], w.box[3]], w.conf)]
        else:
            out.append(w)
    return out


def _classify(t: GraphText, thr: float) -> GraphText:
    s = t.text.strip()
    if apartment_key(s):
        t.kind = TextKind.apartment
    elif parsed := parse_area(s):
        t.kind, (t.value, t.unit) = TextKind.area, parsed
    elif NUMBER_RE.match(s):
        t.kind, t.value = TextKind.dimension, float(s.replace(",", "."))
        if DECIMAL_RE.match(s):
            t.flags.append("number_without_unit")
    elif LABEL_RE.match(s):
        t.kind = TextKind.room_label
    if t.conf is not None and t.conf < thr:
        t.flags.append("low_confidence")
    return t


def _room_at(rooms: list[GraphRoom], polys, p, tol: float) -> GraphRoom | None:
    """Smallest room containing p; tolerance because labels may touch the wall line."""
    d = [cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) for poly in polys]
    inside = [r for r, di in zip(rooms, d) if di >= 0]
    if inside:
        return min(inside, key=lambda r: r.area_px)
    near = [(di, r) for r, di in zip(rooms, d) if di >= -tol]
    return max(near, key=lambda x: x[0])[1] if near else None


def _link(texts: list[GraphText], rooms: list[GraphRoom]) -> None:
    polys = [np.array(r.polygon_px, np.int32).reshape(-1, 1, 2) for r in rooms]
    for t in texts:
        if t.kind != TextKind.area:
            room = _room_at(rooms, polys, _center(t.box_px), tol=max(4, t.box_px[3] - t.box_px[1]))
            t.room_id = room.id if room else None
    labels = [t for t in texts if t.kind == TextKind.room_label and t.room_id]
    for t in (t for t in texts if t.kind == TextKind.area):
        c, h = _center(t.box_px), max(1, t.box_px[3] - t.box_px[1])
        near = [(math.dist(c, _center(l.box_px)), l) for l in labels]
        near = [x for x in near if x[0] < 3 * h]
        if near:  # an area is written right next to its room label: strongest signal
            t.room_id = min(near, key=lambda x: x[0])[1].room_id
        else:
            room = _room_at(rooms, polys, c, tol=max(4, h))
            t.room_id = room.id if room else None


def _finalize(rooms: list[GraphRoom], texts: list[GraphText]) -> None:
    for r in rooms:
        mine = [t for t in texts if t.room_id == r.id]
        labels = sorted((t for t in mine if t.kind == TextKind.room_label),
                        key=lambda t: math.dist(_center(r.box_px), _center(t.box_px)))
        areas = [t for t in mine if t.kind == TextKind.area]
        r.label_text = labels[0].text if labels else None
        r.label_text_ids = [labels[0].id] if labels else []
        r.area_text_ids = [t.id for t in areas]
        values = {round(t.value, 2) for t in areas}
        r.area_m2 = values.pop() if len(values) == 1 else None
        r.issues = (["multiple_area_labels"] if len(values) > 1 else []) + \
                   (["no_area_label"] if not areas else []) + \
                   (["area_low_ocr_confidence"] if any("low_confidence" in t.flags for t in areas) else []) + \
                   ([] if labels else ["no_label_text"])


def apartment_key(text: str) -> str | None:
    """'App. B' -> 'B', 'A' -> 'A'; a bare lower-case 'a' is not a marker."""
    s = text.strip()
    m = APARTMENT_RE.match(s)
    if not m or (m.group(2) and not SINGLE_LETTER_RE.match(s)):
        return None
    return (m.group(1) or m.group(2)).upper()


def _mark_room_numbers(texts: list[GraphText]) -> None:
    """'1.01', '1.02', '1.05' are room numbers, not dimensions: at least MIN_ROOM_NUMBERS distinct numbers
    share the prefix and their suffixes are small. A lone '2.40' stays a dimension / area without unit."""
    cand = [(t, ROOM_NUMBER_RE.match(t.text.strip())) for t in texts if t.kind == TextKind.dimension]
    cand = [(t, m) for t, m in cand if m and int(m.group(2)) <= MAX_ROOM_SUFFIX]
    prefixes = Counter(m.group(1) for m in {m.group(0): m for _, m in cand}.values())
    for t, m in cand:
        if prefixes[m.group(1)] >= MIN_ROOM_NUMBERS:
            t.kind, t.value = TextKind.room_number, None
            t.flags = [f for f in t.flags if f != "number_without_unit"]


def _robust(samples: list[float], src: str) -> dict | None:
    if len(samples) < SCALE_MIN_SAMPLES[src]:
        return None
    med = statistics.median(samples)
    spread = statistics.median(abs(x - med) for x in samples) / med
    if spread > SCALE_MAX_SPREAD:
        return None
    return {"m_per_px": round(med, 6), "src": src, "n": len(samples), "spread": round(spread, 3)}


def _dimension_metres(t: GraphText) -> float | None:
    if t.value is None:
        return None
    decimal = any(c in t.text for c in ".,")
    if not decimal and t.value >= DIM_MM_MIN:
        return t.value / 1000
    if decimal and t.value < DIM_M_MAX:
        return t.value
    return None


def _dimension_samples(texts: list[GraphText], walls: list[GraphWall]) -> list[float]:
    """A dimension written along a wall run measures (about) that run: metres / run length in px."""
    out = []
    for t in texts:
        metres = _dimension_metres(t) if t.kind == TextKind.dimension else None
        if not metres:
            continue
        x0, y0, x1, y1 = t.box_px
        orient = "h" if (x1 - x0) >= (y1 - y0) else "v"
        cx, cy = _center(t.box_px)
        reach = 4 * max(1, min(x1 - x0, y1 - y0))
        near = []
        for w in walls:
            if w.kind != "wall" or w.orientation != orient:
                continue
            wx0, wy0, wx1, wy1 = w.box_px
            if orient == "h" and wx0 <= cx <= wx1:
                near.append((abs(cy - (wy0 + wy1) / 2), w))
            elif orient == "v" and wy0 <= cy <= wy1:
                near.append((abs(cx - (wx0 + wx1) / 2), w))
        near = [x for x in near if x[0] <= reach]
        if near:
            out.append(metres / min(near, key=lambda x: x[0])[1].length_px)
    return out


def estimate_scale(rooms: list[GraphRoom], texts: list[GraphText], walls: list[GraphWall]) -> dict | None:
    """Metres per pixel. First choice: rooms with a written area (sqrt(m2 / px2), consistent across rooms);
    otherwise dimensions next to wall runs. None when the samples disagree (no scale beats a wrong one)."""
    by_area = [math.sqrt(r.area_m2 / r.area_px) for r in rooms if r.area_m2 and r.area_px > 0]
    return _robust(by_area, "room_areas") or _robust(_dimension_samples(texts, walls), "dimensions")
