"""Plan graph (OCR + layout model on one drawing) -> common building-graph/1.

Ids stay those of plan_graph.json (r* rooms, t* texts, o* symbols) so the report can
highlight what the agent cites. Everything that is assumed rather than detected is marked
(src=inferred, issues, notes), so it stays visible in the explanation.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Union

from .building_schema import (OUTDOOR, BuildingGraph, Extraction, Source, Storey, compute_facts, fold,
                              fuzzy_categories, make_table, space_categories, space_use)
from .qa import FLOOR_LABEL, MODEL_CLASS
from .schemas import PlanGraph, TextKind

ROOM_TYPE_CATEGORY = {"bathroom": "bathroom", "sauna": "sauna", "kitchen": "kitchen", "living_room": "living",
                      "bedroom": "bedroom", "hallway": "hallway", "storage": "storage", "garage": "garage",
                      "balcony": "balcony"}
WET = {"bathroom", "wc", "sauna"}

OPENING_MARK = re.compile(r"\d{1,2}\s*[x×]\s*\d{1,2}\b")
SPACE_COLS = ["id", "name", "storey", "categories", "use", "outdoor", "net_area", "conf", "src", "model_class",
              "label_ids", "area_ids", "issues", "box_px"]
OBJ_COLS = ["id", "storey", "space", "category", "conf", "src", "box_px"]
TEXT_COLS = ["id", "text", "kind", "conf", "space", "value", "unit", "flags", "box_px"]


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


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
    for t in p.texts:
        if t.room_id and t.kind not in (TextKind.area, TextKind.dimension):
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
        rows.append([r.id, name, storey_of(r.box_px), cats or None, space_use(cats), outdoor or None,
                     r.area_m2, round(conf, 2), src, r.model_class, r.label_text_ids or None, r.area_text_ids or None,
                     issues or None, r.box_px])
    for t, c, why, rid in second:
        cats_of[t.id] = c
        rows.append([t.id, t.text, storey_of(t.box_px), c, space_use(c), bool(set(c) & OUTDOOR) or None, None, 0.6,
                     "room_label", None, [t.id], None, [f"second_label_in_room:{rid}"] + ([why] if why else []), t.box_px])
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
                             ["label_without_detected_room"] + ([why] if why else []), t.box_px])
    if orphans:
        notes.append(f"{orphans} room label(s) lie outside every detected room: listed as spaces with "
                     "src=room_label and issue label_without_detected_room (no outline, no area).")

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
            groups["furniture"].append(base + [o.kind, o.conf, "symbol", o.box_px])
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

    thr = p.min_conf if isinstance(p.min_conf, dict) else {"all": p.min_conf}
    if any(p.dropped.values()):
        notes.append("Detections below the confidence thresholds were dropped (" +
                     ", ".join(f"{k} {v}" for k, v in p.dropped.items()) + "): counts can be too low.")
    if not any(r.area_m2 for r in p.rooms):
        notes.append("No room area is written in the drawing (or none was read): areas are unknown.")
    else:
        notes.append("Room areas are only those written in the drawing (net_area); rooms without an area label have none.")
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
    how = {"room_label": "label", "room_text": "text in the room", "room_polygon": "layout model class"}

    def desc(s):
        extra = [i for i in (s.get("issues") or []) if i.startswith(("ocr_fuzzy", "second_label", "label_without",
                                                                      "reserved"))]
        return f"{s['id']}" + (f" \"{s['name']}\"" if s.get("name") else "") + f" ({how.get(s.get('src'), s.get('src'))}" \
            + (", " + ", ".join(extra) if extra else "") + ")"
    skip_issue = "unlabelled_outdoor" if cat == "balcony" else None
    skipped = [s for s in sp if skip_issue and any(i.startswith(skip_issue) for i in (s.get("issues") or []))]
    conf = round(sum(s.get("conf") or 0.6 for s in sel) / len(sel), 2) if sel else 0.6
    expl = (f"{len(sel)} counted: " + "; ".join(desc(s) for s in sel)) if sel else \
        f"No room of category '{cat}' found (neither by label/text nor by the layout model)"
    if skipped:
        expl += "; not counted: " + ", ".join(
            f"{s['id']} ({next(i for i in s['issues'] if i.startswith(skip_issue))})" for s in skipped)
    return Answer(intent, len(sel), confidence=conf, evidence=[s["id"] for s in sel], explanation=expl + ".")
