"""Plan graph (OCR + layout model on one drawing) -> common building-graph/1.

Ids stay those of plan_graph.json (r* rooms, t* texts, o* symbols) so the report can
highlight what the agent cites. Everything that is assumed rather than detected is marked
(src=inferred, issues, notes), so it stays visible in the explanation.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Union

from .building_schema import (OUTDOOR, BuildingGraph, Extraction, Source, Storey, compute_facts, fold,
                              fuzzy_categories, make_table, space_categories, space_use)
from .build_graph import apartment_key
from .qa import FLOOR_LABEL, MODEL_CLASS
from .schemas import PlanGraph, TextKind

ROOM_TYPE_CATEGORY = {"bathroom": "bathroom", "sauna": "sauna", "kitchen": "kitchen", "living_room": "living",
                      "bedroom": "bedroom", "hallway": "hallway", "storage": "storage", "garage": "garage",
                      "balcony": "balcony"}
WET = {"bathroom", "wc", "sauna"}

OPENING_MARK = re.compile(r"\d{1,2}\s*[x×]\s*\d{1,2}\b")
SPACE_COLS = ["id", "name", "storey", "categories", "use", "outdoor", "net_area", "conf", "src", "model_class",
              "label_ids", "area_ids", "issues", "box_px",
              "number", "unit", "area_src", "area_px", "model_type", "model_conf", "type_reason", "polygon_px"]
SC = {c: i for i, c in enumerate(SPACE_COLS)}
OBJ_COLS = ["id", "storey", "space", "category", "conf", "src", "box_px"]
WALL_COLS = ["id", "storey", "category", "orientation", "length", "thickness", "length_px", "thickness_px", "conf",
             "src", "box_px"]
OBJECT_CATEGORY = {"electrical_appliance": "appliance"}   # symbol kind -> furniture category
NEXT_TO_LABEL = 3   # an area / room number within 3 label heights belongs to that label
UNIT_EDGE_TOL = 3   # a label outside every region within 3 label heights of a marker's region joins it
TEXT_COLS = ["id", "text", "kind", "conf", "space", "value", "unit", "flags", "box_px"]


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _poly_distance(row, p) -> float:
    """Distance of point p to a space: 0 inside its outline, else to the outline (or box centre)."""
    poly = row[SC["polygon_px"]]
    if poly:
        import cv2
        import numpy as np
        d = cv2.pointPolygonTest(np.array(poly, np.int32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), True)
        return max(0.0, -d)
    return math.dist(_center(row[SC["box_px"]]), p)


def _assign_units(rows, markers, texts, notes) -> None:
    """Apartment of each space from the apartment markers ('A', 'App. B').

    region mode (some marker lies inside a detected layout region): a space belongs to the marker of the region it
    lies in, or whose edge its label touches (issue unit_near_marker_region). Spaces elsewhere (shared corridor,
    cellar storage, legend) get no unit: better unknown than added to the wrong apartment.
    nearest mode (no marker inside any region): the nearest marker on the same storey (issue unit_from_nearest_marker)."""
    if not markers:
        return
    region_unit = {m[3]: m[0] for m in markers if m[3]}
    text_room = {t.id: t.room_id for t in texts}
    text_box = {t.id: t.box_px for t in texts}
    by_id = {row[0]: row for row in rows}
    if region_unit:
        for row in rows:
            rid = row[0] if row[0] in region_unit else text_room.get(row[0])
            if rid in region_unit:
                row[SC["unit"]] = region_unit[rid]
                continue
            if rid and not row[SC["polygon_px"]]:
                continue    # second label inside another region without a marker
            # the space's label just across the edge of a marker region (a bath split off from its apartment)
            box = text_box.get((row[SC["label_ids"]] or [None])[0]) or (None if row[SC["polygon_px"]] else row[SC["box_px"]])
            if box is None:
                continue    # unlabelled region without a marker: unknown
            label_h = max(1, box[3] - box[1])
            near = [(_poly_distance(by_id[r], _center(box)), k) for r, k in region_unit.items()
                    if r in by_id and r != row[0]]
            near = [n for n in near if n[0] <= UNIT_EDGE_TOL * label_h]
            if near:
                row[SC["unit"]] = min(near)[1]
                row[SC["issues"]] = (row[SC["issues"]] or []) + ["unit_near_marker_region"]
        how = ("a space belongs to the marker of the detected region it lies in (or whose edge its label touches, "
               "issue unit_near_marker_region); spaces outside those regions (shared corridors, storage) have no unit")
    else:
        for row in rows:
            same = [m for m in markers if row[2] is None or m[2] in (None, row[2])]
            if not same:
                continue
            dist, key = min((_poly_distance(row, m[1]), m[0]) for m in same)
            row[SC["unit"]] = key
            if dist > 0:
                row[SC["issues"]] = (row[SC["issues"]] or []) + ["unit_from_nearest_marker"]
        how = ("no marker lies inside a detected region, so every space belongs to the nearest marker (issue "
               "unit_from_nearest_marker; an estimate, walls between apartments are not checked)")
    keys = sorted({m[0] for m in markers})
    notes.append(f"Apartments from {len(markers)} apartment marker(s) written in the plan ({', '.join(keys)}): {how}.")


def _pair_with_label(rows, texts, notes) -> None:
    """Plans often write label, room number and area stacked ('Pièce' / '5.02' / '11.71 m²'). When the layout
    merged several rooms (or found none), the room polygon cannot tell which area is whose: the text written
    right under a space's own label can. Areas already backing a room's single written area stay with it."""
    by_id = {t.id: t for t in texts}
    used = {i for row in rows if row[SC["net_area"]] is not None for i in (row[SC["area_ids"]] or [])}
    paired = 0

    def next_to(label, kind):
        c, h = _center(label.box_px), max(1, label.box_px[3] - label.box_px[1])
        near = [(math.dist(c, _center(x.box_px)), x) for x in texts if x.kind == kind and x.id not in used]
        near = [n for n in near if n[0] < NEXT_TO_LABEL * h]
        if not near:
            return None
        hit = min(near, key=lambda n: n[0])[1]
        used.add(hit.id)
        return hit

    for row in rows:
        label = by_id.get((row[SC["label_ids"]] or [None])[0])
        if label is None:
            continue
        num = next_to(label, TextKind.room_number)
        if num:
            row[SC["number"]] = num.text.strip()
        if row[SC["net_area"]] is None:
            area = next_to(label, TextKind.area)
            if area and area.value is not None:
                row[SC["net_area"]], row[SC["area_ids"]], row[SC["area_src"]] = round(area.value, 2), [area.id], "label"
                row[SC["issues"]] = (row[SC["issues"]] or []) + [f"area_next_to_label:{area.id}"]
                paired += 1
    if paired:
        notes.append(f"{paired} area(s) taken from the text written right next to the space's label (issue "
                     "area_next_to_label), because the room outline held several areas or there was no outline.")


def to_building_graph(plan: Union[PlanGraph, dict]) -> BuildingGraph:
    p = plan if isinstance(plan, PlanGraph) else PlanGraph.model_validate(plan)
    notes: list[str] = []

    # ---- storeys: drawn floor outlines, or written storey names if there are more of them
    labels = [t for t in p.texts if FLOOR_LABEL.search(t.text)]
    if len(labels) >= 2 and len(labels) >= len(p.floors):
        storeys = [Storey(id=f"L{i}", name=t.text) for i, t in enumerate(labels)]
        notes.append(f"Storeys from {len(labels)} storey names written in the plan ({', '.join(t.id for t in labels)}).")
    else:
        boxes = p.floors or [[0, 0, *p.image_size]]
        storeys = [Storey(id=f"L{i}", name=f"floor outline {i + 1}", box_px=b) for i, b in enumerate(boxes)]
        if not p.floors:
            notes.append("No floor outline detected: the whole drawing is treated as one storey.")

    def storey_of(box):
        if len(storeys) == 1:
            return storeys[0].id
        x, y = _center(box)
        for s in storeys:
            b = s.box_px
            if b and b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                return s.id
        return None

    # ---- spaces: detected rooms + room labels that lie outside every detected room
    def cats_of_text(text):
        """(categories, issue) of a written label: exact (abbreviation / word) first, then OCR-typo tolerant."""
        c = space_categories(text)
        if c:
            return c, None
        c, word = fuzzy_categories(text)
        return (c, f"ocr_fuzzy:{word}") if c else ([], None)

    texts_in: dict[str, list] = {}
    numbers_in: dict[str, list] = {}
    for t in p.texts:
        if t.room_id and t.kind == TextKind.room_number:
            numbers_in.setdefault(t.room_id, []).append(t)
        elif t.room_id and t.kind not in (TextKind.area, TextKind.dimension, TextKind.apartment):
            texts_in.setdefault(t.room_id, []).append(t)
    outdoor_labelled = any(set(cats_of_text(x)[0]) & OUTDOOR
                           for x in [r.label_text for r in p.rooms] + [t.text for t in p.texts
                                                                        if t.kind == TextKind.room_label])
    rows, cats_of, second = [], {}, []
    for r in p.rooms:
        # candidates: the chosen label, other room labels in the room, then other texts (e.g. 'OH+RT+K', 'MH4(VARAUS)')
        cand = sorted(texts_in.get(r.id, []), key=lambda t: (t.id not in r.label_text_ids, t.kind != TextKind.room_label))
        if r.label_text and not any(t.text == r.label_text for t in cand):   # the room's own label comes first
            cand.insert(0, SimpleNamespace(id=(r.label_text_ids or [r.id])[0], text=r.label_text, kind=TextKind.room_label))
        cats, name, src, conf, issues, used = [], r.label_text, "room_polygon", r.type_conf, list(r.issues), None
        for t in cand:
            c, why = cats_of_text(t.text)
            if c:
                cats, name, used = c, t.text, t
                src = "room_label" if t.kind == TextKind.room_label else "room_text"
                conf = 0.9 if src == "room_label" and not why else 0.75 if not why else 0.7
                issues += [why] if why else []
                if t.text != r.label_text:
                    issues.append(f"label_from_text:{t.id}")
                break
        for t in cand:   # a second room label of another kind: the layout merged two rooms (e.g. sauna 'S' in a bath)
            if t is not used and t.kind == TextKind.room_label and t.text != name:
                c, why = cats_of_text(t.text)
                if c and not set(c) & set(cats):
                    second.append((t, c, why, r.id))
        if not cats:
            c = "sauna" if r.room_type.value == "sauna" else MODEL_CLASS.get(r.model_class) \
                or ROOM_TYPE_CATEGORY.get(r.room_type.value)
            cats = [c] if c else []
        if name and "VARAUS" in fold(name):
            issues.append("reserved_room_VARAUS")
        outdoor = bool(set(cats) & OUTDOOR) or r.model_class == "Outdoor"
        if src == "room_polygon" and "balcony" in cats and outdoor_labelled:
            cats = []   # 'Outdoor' also covers terraces/yards; the plan names its balconies, this one has no label
            issues.append("unlabelled_outdoor_not_counted_as_balcony")
        cats_of[r.id] = cats
        number = next((t.text.strip() for t in sorted(numbers_in.get(r.id, []),
                                                       key=lambda t: math.dist(_center(r.box_px), _center(t.box_px)))),
                      None)
        # the label the name came from (not just the one nearest the centre): area / number are paired with it
        label_ids = [used.id] if used is not None and str(used.id).startswith("t") else (r.label_text_ids or None)
        rows.append([r.id, name, storey_of(r.box_px), cats or None, space_use(cats), outdoor or None,
                     r.area_m2, round(conf, 2), src, r.model_class, label_ids, r.area_text_ids or None,
                     issues or None, r.box_px,
                     number, None, "label" if r.area_m2 is not None else None, r.area_px, r.room_type.value,
                     r.type_conf, r.type_reason or None, r.polygon_px])
    for t, c, why, rid in second:
        cats_of[t.id] = c
        rows.append([t.id, t.text, storey_of(t.box_px), c, space_use(c), bool(set(c) & OUTDOOR) or None, None, 0.6,
                     "room_label", None, [t.id], None, [f"second_label_in_room:{rid}"] + ([why] if why else []), t.box_px,
                     None, None, None, None, None, None, None, None])
    if second:
        notes.append(f"{len(second)} room(s) from a second room label inside a detected room (the layout model merged "
                     "two rooms): issue second_label_in_room.")

    if any("ocr_fuzzy" in str(row[12]) for row in rows):
        notes.append("Some labels were read with one or two wrong letters (OCR) and matched to the closest room word "
                     "(issue ocr_fuzzy:<word>).")
    if any("VARAUS" in str(row[12]) for row in rows):
        notes.append("Rooms marked VARAUS (reserved, e.g. 'MH4 (VARAUS)') are counted as that room type.")
    orphans = 0
    for t in p.texts:
        if t.kind == TextKind.room_label and t.room_id is None:
            cats, why = cats_of_text(t.text)
            if cats:
                orphans += 1
                cats_of[t.id] = cats
                rows.append([t.id, t.text, storey_of(t.box_px), cats, space_use(cats), bool(set(cats) & OUTDOOR) or None,
                             None, 0.6, "room_label", None, [t.id], None,
                             ["label_without_detected_room"] + ([why] if why else []), t.box_px,
                             None, None, None, None, None, None, None, None])
    if orphans:
        notes.append(f"{orphans} room label(s) lie outside every detected room: listed as spaces with "
                     "src=room_label and issue label_without_detected_room (no outline; net_area / number only when "
                     "written right next to the label).")

    _pair_with_label(rows, p.texts, notes)
    if any(row[SC["number"]] for row in rows):
        notes.append("Room numbers (e.g. 1.04) are OCR texts inside the room outline: spaces[number].")
    markers = [(apartment_key(t.text), _center(t.box_px), storey_of(t.box_px), t.room_id) for t in p.texts
               if t.kind == TextKind.apartment]
    _assign_units(rows, markers, p.texts, notes)

    scale = p.scale
    if scale:
        k = scale["m_per_px"] ** 2
        est = 0
        for row in rows:
            if row[SC["net_area"]] is None and row[SC["area_px"]]:
                row[SC["net_area"]] = round(row[SC["area_px"]] * k, 2)
                row[SC["area_src"]] = f"estimate:{scale['src']}"
                est += 1
        how = ("rooms whose written area matches their pixel area" if scale["src"] == "room_areas"
               else "dimensions written along wall runs")
        notes.append(f"Scale {scale['m_per_px']} m/px from {scale['n']} {how} (spread {scale['spread']}). "
                     f"{est} room area(s) without a written area are estimated from the pixel area "
                     "(area_src=estimate:..., call them estimates); area_src=label is written in the drawing.")


    # ---- elements: symbols of the layout model
    groups: dict[str, list] = {"doors": [], "windows": [], "sanitary": [], "furniture": []}
    tubs_in = set()
    for o in p.objects:
        base = [o.id, storey_of(o.box_px), o.room_id]
        if o.kind in ("door", "window"):
            groups[o.kind + "s"].append(base + [None, o.conf, "symbol", o.box_px])
        elif o.kind in ("toilet", "sink", "bathtub"):
            cat = o.kind
            if o.kind == "sink":
                cat = "washbasin" if set(cats_of.get(o.room_id, [])) & WET else "sink"
            if o.kind == "bathtub":
                tubs_in.add(o.room_id)
            groups["sanitary"].append(base + [cat, o.conf, "symbol", o.box_px])
        else:
            groups["furniture"].append(base + [OBJECT_CATEGORY.get(o.kind, o.kind), o.conf, "symbol", o.box_px])
    has_layout = bool(p.rooms or p.objects)
    if has_layout:
        for row in rows:
            if "bathroom" in (row[3] or []) and row[0] not in tubs_in:
                groups["sanitary"].append([f"{row[0]}.shower", row[2], row[0], "shower", 0.5, "inferred", None])
        if any(r[5] == "inferred" for r in groups["sanitary"]):
            notes.append("The layout model has no shower symbol: one shower is inferred per bathroom without a "
                         "bathtub (sanitary rows with src=inferred, id <room>.shower).")
        if any(r[3] in ("sink", "washbasin") for r in groups["sanitary"]):
            notes.append("Sink vs. washbasin is decided by the room the symbol lies in (bathroom/WC/sauna -> washbasin).")
    else:
        notes.append("Layout model did not run: no rooms, doors, windows or fixtures, only OCR texts.")
    elements = {g: make_table(OBJ_COLS, v, ("id", "storey", "space", "conf") + (("category",) if g in ("sanitary", "furniture")
                                                                               else ())) for g, v in groups.items()}
    m_per_px = scale["m_per_px"] if scale else None
    wall_rows = [[w.id, storey_of(w.box_px), w.kind, w.orientation,
                  round(w.length_px * m_per_px, 2) if m_per_px else None,
                  round(w.thickness_px * m_per_px, 3) if m_per_px else None,
                  w.length_px, w.thickness_px, w.conf, "layout_class", w.box_px] for w in p.walls]
    if wall_rows:
        elements["walls"] = make_table(WALL_COLS, wall_rows, ("id", "storey", "category", "length_px", "conf"))
        notes.append(f"elements.walls: {len(wall_rows)} straight runs of the layout model's Wall / Railing class "
                     "(one row per horizontal or vertical run, not per modelled wall; diagonal walls are missing)"
                     + ("; length / thickness in m via the plan scale." if m_per_px else
                        "; no scale, so only pixel lengths."))
    rejected = [[o.id, storey_of(o.box_px), o.room_id, o.kind, o.conf, "below_threshold", o.box_px]
                for o in p.rejected_objects] + \
               [[r.id, storey_of(r.box_px), None, f"room:{r.model_class}", r.type_conf, "below_threshold", r.box_px]
                for r in p.rejected_rooms]
    if rejected:
        elements["rejected"] = make_table(OBJ_COLS, rejected, ("id", "category", "conf"))
        notes.append(f"elements.rejected: {len(rejected)} detection(s) below the confidence thresholds (ids ox*, rx*). "
                     "They are NOT counted; use them only to explain why a count may be too low.")

    def ann_kind(t):
        if t.kind == TextKind.other and len(t.text) <= 20 and OPENING_MARK.search(t.text):
            return "opening_size"   # '9x21' door / 'A/12x14' window size mark
        return t.kind.value
    ann = [[t.id, t.text, ann_kind(t), None if t.conf is None else round(t.conf, 3), t.room_id, t.value, t.unit,
            t.flags or None, t.box_px] for t in p.texts]

    # ---- storey aggregates
    st = {s.id: s for s in storeys}
    for row in rows:
        s = st.get(row[2])
        if s:
            s.n_spaces += 1
            if row[6] is not None and not row[5]:
                s.net_floor_area = round(s.net_floor_area + row[6], 2)
    for g, v in groups.items():
        for row in v:
            if row[1] in st:
                st[row[1]].counts[g] = st[row[1]].counts.get(g, 0) + 1
    for row in wall_rows:
        s = st.get(row[1])
        if s and row[2] == "wall" and row[4] is not None:
            s.walls["length"] = round(s.walls.get("length", 0.0) + row[4], 2)

    thr = p.min_conf if isinstance(p.min_conf, dict) else {"all": p.min_conf}
    if any(p.dropped.values()):
        notes.append("Detections below the confidence thresholds were dropped (" +
                     ", ".join(f"{k} {v}" for k, v in p.dropped.items()) + "): counts can be too low.")
    if not any(r.area_m2 for r in p.rooms) and not scale:
        notes.append("No room area is written in the drawing (or none was read) and no scale: areas are unknown.")
    elif not scale:
        notes.append("Room areas are only those written in the drawing (net_area); rooms without an area label have none "
                     "(no consistent scale to estimate them).")
    kind = "pdf" if Path(p.source).suffix.lower() == ".pdf" else "image"
    g = BuildingGraph(
        source=Source(file=p.source, kind=kind, page=p.page, image_size=p.image_size,
                      extraction=Extraction(method="ocr+layout" if has_layout else "ocr",
                                            models={"ocr": "PaddleOCR", "layout": "CubiCasa5K"} if has_layout
                                            else {"ocr": "PaddleOCR"},
                                            thresholds=thr, dropped=p.dropped)),
        storeys=storeys, spaces=make_table(SPACE_COLS, rows, ("id", "name", "storey", "categories", "net_area", "conf")),
        elements=elements, annotations=make_table(TEXT_COLS, ann, ("id", "text", "kind", "conf", "space")),
        notes=notes)
    g.facts = compute_facts(g)
    return g


def as_building_graph(data: dict) -> dict:
    """Any supported JSON -> building-graph dict (plan_graph.json is converted, others pass through)."""
    if "rooms" in data and "texts" in data and "schema_version" not in data:
        return to_building_graph(data).model_dump(mode="json", exclude_none=True)
    return data


ROOM_INTENTS = {"bedrooms": "bedroom", "kitchens": "kitchen", "bathrooms": "bathroom", "wc_rooms": "wc",
                "saunas": "sauna", "balconies": "balcony"}


def room_answer(g: BuildingGraph, intent: str):
    """Rule answer for a room-count question from the building graph (same assignment as facts):
    every counted space with how it was recognised, plus the parts that were deliberately not counted."""
    from .qa import Answer
    cat = ROOM_INTENTS[intent]
    sp = [dict(zip(g.spaces.columns, r)) for r in g.spaces.rows]
    sel = [s for s in sp if cat in (s.get("categories") or [])]
    how = {"room_label": "Label", "room_text": "Text im Raum", "room_polygon": "Klasse des Layout-Modells"}

    def desc(s):
        extra = [i for i in (s.get("issues") or []) if i.startswith(("ocr_fuzzy", "second_label", "label_without",
                                                                      "reserved"))]
        return f"{s['id']}" + (f" „{s['name']}“" if s.get("name") else "") + f" ({how.get(s.get('src'), s.get('src'))}" \
            + (", " + ", ".join(extra) if extra else "") + ")"
    skip_issue = "unlabelled_outdoor" if cat == "balcony" else None
    skipped = [s for s in sp if skip_issue and any(i.startswith(skip_issue) for i in (s.get("issues") or []))]
    conf = round(sum(s.get("conf") or 0.6 for s in sel) / len(sel), 2) if sel else 0.6
    expl = (f"{len(sel)} gezählt: " + "; ".join(desc(s) for s in sel)) if sel else \
        f"Kein Raum der Kategorie '{cat}' gefunden (weder per Label/Text noch per Layout-Modell)"
    if skipped:
        expl += "; nicht gezählt: " + ", ".join(
            f"{s['id']} ({next(i for i in s['issues'] if i.startswith(skip_issue))})" for s in skipped)
    return Answer(intent, len(sel), confidence=conf, evidence=[s["id"] for s in sel], explanation=expl + ".")
