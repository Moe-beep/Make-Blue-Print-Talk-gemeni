"""Build explanation.json (planqa/explanation_schema.py) for a floor plan from data the pipeline already has.

Inputs (all from one answer.py run, no extra model calls, no Decisions API yet):
  graph       PlanGraph (what the answers use; detections below the thresholds are not in it)
  detections  raw model output (Pipeline.last_detections): all OCR words, rooms, symbols incl. low confidence
  probs       (room_prob [12,H,W], icon_prob [11,H,W]) of the layout model (Pipeline.last_probs)
  records     answer records of answer.py (rules + LLM agent incl. trace)

Outputs: ExplanationDoc + heat-map images per channel (uncertainty = normalised entropy of the
class probabilities; for OCR 1 - confidence inside each word box).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .explanation_schema import (Answer, Assets, BlindSpot, ClassProb, Coverage, Element, Evidence,
                                 ExplanationDoc, Geometry, Hotspot, Run, Source, SourceAnswer, Step)

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath",
                "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
ICON_CLASSES = ["Empty", "Window", "Door", "Closet", "Electrical appliance", "Toilet", "Sink",
                "Sauna bench", "Fire place", "Bathtub", "Chimney"]
INTENT_KINDS = {"doors": {"door"}, "windows": {"window"}, "toilets": {"toilet"}, "sinks": {"sink"},
                "showers_bathtubs": {"bathtub"}, "saunas": {"sauna_bench"}}
ROOM_INTENTS = {"bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas", "balconies"}
KIND_DE = {"door": "door", "window": "window", "toilet": "toilet", "sink": "sink", "bathtub": "bathtub",   # UI labels (English)
           "sauna_bench": "sauna bench", "closet": "closet", "room": "room", "text": "text"}
INTENT_DE = {"doors": "count doors", "windows": "count windows", "toilets": "count toilets",   # UI labels (English)
             "sinks": "count sinks", "showers_bathtubs": "count showers/bathtubs", "saunas": "count saunas",
             "bathrooms": "count bathrooms", "wc_rooms": "count WC rooms", "kitchens": "count kitchens",
             "bedrooms": "count bedrooms", "balconies": "count balconies", "floors": "count storeys"}
TABLE_KINDS = {"elements.doors": {"door"}, "elements.windows": {"window"},
               "elements.sanitary": {"toilet", "sink", "bathtub", "sauna_bench"}, "elements.furniture": {"closet"}}
ID_RE = re.compile(r"\b([rot]\d{2,4})\b")


# ============================================================================ uncertainty maps

def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0)
    return (-(p * np.log(p)).sum(0) / np.log(p.shape[0])).astype(np.float32)


def uncertainty_maps(size, probs, words) -> dict[str, np.ndarray]:
    """Per-pixel uncertainty in [0, 1] per channel (rooms, symbols, text) and combined (max)."""
    W, H = size
    maps = {}
    if probs is not None:
        room_prob, icon_prob = probs
        maps["rooms"] = _entropy(room_prob)
        maps["symbols"] = _entropy(icon_prob)
    text = np.zeros((H, W), np.float32)
    for w in words or []:
        x0, y0, x1, y1 = [int(round(v)) for v in w.box]
        text[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = np.maximum(
            text[max(0, y0):max(0, y1), max(0, x0):max(0, x1)], 1.0 - float(w.conf))
    maps["text"] = text
    maps["combined"] = np.maximum.reduce([maps[k] for k in ("rooms", "symbols", "text") if k in maps])
    return maps


_STOPS = [(0.0, (255, 250, 205)), (0.3, (254, 178, 76)), (0.55, (240, 59, 32)), (0.8, (128, 0, 90)), (1.0, (40, 0, 50))]


def heat_rgba(u: np.ndarray, max_side: int = 1600) -> Image.Image:
    """Transparent where certain, yellow -> red -> purple where uncertain."""
    xs = [s[0] for s in _STOPS]
    rgb = np.stack([np.interp(u, xs, [s[1][c] for s in _STOPS]) for c in range(3)], -1)
    a = np.clip((u - 0.15) * 1.4, 0, 0.88) * 255
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    s = min(1.0, max_side / max(img.size))
    return img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR) if s < 1 else img


# ============================================================================ elements

def _box_of(poly):
    a = np.asarray(poly, float)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def _top(probs_mean: np.ndarray, names, k=3):
    idx = np.argsort(-probs_mean)[:k]
    return [ClassProb(label=names[i], p=round(float(probs_mean[i]), 3)) for i in idx if probs_mean[i] > 0.01]


def _mean_in_box(prob, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    sub = prob[:, max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    return sub.reshape(sub.shape[0], -1).mean(1) if sub.size else np.zeros(prob.shape[0])


def _mean_in_poly(prob, poly):
    x0, y0, x1, y1 = [int(round(v)) for v in _box_of(poly)]
    x0, y0 = max(0, x0), max(0, y0)
    m = np.zeros((max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly) - [x0, y0]).astype(np.int32)], 1)
    sub = prob[:, y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    mm = m[:sub.shape[1], :sub.shape[2]] > 0
    return sub[:, mm].mean(1) if mm.any() else np.zeros(prob.shape[0])


def build_elements(graph, det, probs, th) -> dict[str, Element]:
    """Kept elements under their graph ids + candidates below the thresholds (x*), which the
    answers never saw. Class probabilities of the layout model are added where available."""
    room_prob, icon_prob = probs if probs is not None else (None, None)
    els: dict[str, Element] = {}
    for r in graph.rooms:
        cp = _top(_mean_in_poly(room_prob, r.polygon_px), ROOM_CLASSES) if room_prob is not None and r.polygon_px else []
        lab = f"{r.label_text} ({r.room_type})" if r.label_text else str(r.room_type)
        els[r.id] = Element(id=r.id, kind="room", label=lab, conf=r.type_conf, class_probs=cp, status="kept",
                            geometry=Geometry(polygon=r.polygon_px or None, bbox=list(map(float, r.box_px))))
    for o in graph.objects:
        cp = _top(_mean_in_box(icon_prob, o.box_px), ICON_CLASSES) if icon_prob is not None else []
        els[o.id] = Element(id=o.id, kind=o.kind, label=KIND_DE.get(o.kind, o.kind), conf=o.conf, class_probs=cp,
                            status="kept", room=o.room_id, geometry=Geometry(bbox=list(map(float, o.box_px))))
    for t in graph.texts:
        els[t.id] = Element(id=t.id, kind="text", label=t.text, conf=t.conf, status="kept", room=t.room_id,
                            geometry=Geometry(bbox=list(map(float, t.box_px))))
    if det:   # candidates the thresholds removed
        n = 0
        for o in det.get("objects") or []:
            if o.conf < th["objects"]:
                n += 1
                cp = _top(_mean_in_box(icon_prob, o.box_px), ICON_CLASSES) if icon_prob is not None else []
                els[f"x{n:03d}"] = Element(id=f"x{n:03d}", kind=o.kind, label=KIND_DE.get(o.kind, o.kind), conf=o.conf,
                                           class_probs=cp, status="below_threshold",
                                           geometry=Geometry(bbox=list(map(float, o.box_px))))
        for r in det.get("rooms") or []:
            if r.type_conf < th["rooms"]:
                n += 1
                els[f"x{n:03d}"] = Element(id=f"x{n:03d}", kind="room", label=f"{r.model_class}", conf=r.type_conf,
                                           status="below_threshold",
                                           geometry=Geometry(polygon=r.polygon_px or None, bbox=list(map(float, r.box_px))))
        for w in det.get("words") or []:
            if w.conf < th["texts"]:
                n += 1
                els[f"x{n:03d}"] = Element(id=f"x{n:03d}", kind="text", label=w.text, conf=round(float(w.conf), 3),
                                           status="below_threshold",
                                           geometry=Geometry(bbox=[float(v) for v in w.box]))
    return els


# ============================================================================ hotspots, blind spots

def find_hotspots(maps, probs, els, th, max_n=8, thr=0.45):
    """Connected uncertain regions (smoothed, so thin wall/room borders drop out), ranked."""
    comb = maps["combined"]
    H, W = comb.shape
    k = max(9, int(0.012 * max(H, W)) | 1)
    sm = cv2.blur(comb, (k, k))
    mask = (sm > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(30, int(0.0003 * H * W))
    cands = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        comp = lab[y:y + h, x:x + w] == i
        score = float(sm[y:y + h, x:x + w][comp].mean())
        means = {c: float(maps[c][y:y + h, x:x + w][comp].mean()) for c in ("rooms", "symbols", "text") if c in maps}
        cands.append((score * np.log1p(a), score, i, (x, y, w, h), comp, means))
    cands.sort(key=lambda c: -c[0])
    room_prob, icon_prob = probs if probs is not None else (None, None)
    hots = []
    for rank, (_, score, i, (x, y, w, h), comp, means) in enumerate(cands[:max_n], 1):
        ch = max(means, key=means.get)
        box = [float(x), float(y), float(x + w), float(y + h)]
        inside = [e for e in els.values() if e.geometry.bbox and _overlap(e.geometry.bbox, box) > 0.2]
        if ch == "rooms" and room_prob is not None:
            top = _top(room_prob[:, y:y + h, x:x + w][:, comp].mean(1), ROOM_CLASSES)
        elif ch == "symbols" and icon_prob is not None:
            top = _top(icon_prob[:, y:y + h, x:x + w][:, comp].mean(1), ICON_CLASSES)
        else:
            top = [ClassProb(label=f"'{e.label}'", p=round(e.conf or 0, 3)) for e in inside if e.kind == "text"][:3]
        rel = [e for e in inside if (ch == "rooms") == (e.kind == "room") and (ch == "text") == (e.kind == "text")] or inside
        res, by = _resolution(rel, th)
        hots.append(Hotspot(id=f"h{rank}", rank=rank, entropy=round(score, 3), channel=ch, geometry=Geometry(bbox=box),
                            top_classes=top, elements=[e.id for e in rel][:6], resolution=res, resolved_by=by))
    return hots


def _overlap(a, b):
    """Share of box a that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def _resolution(rel, th):
    for e in rel:
        if e.kind == "room" and e.status == "kept" and "(" in (e.label or ""):
            return f"OCR label '{e.label.split(' (')[0]}' decides: {e.label.split('(')[-1].rstrip(')')}", "ocr_label"
    kept = [e for e in rel if e.status == "kept"]
    if kept:
        e = kept[0]
        return f"kept: {e.label} (confidence {e.conf:.2f})", "rule"
    low = [e for e in rel if e.status == "below_threshold"]
    if low:
        e = low[0]
        t = th["texts" if e.kind == "text" else "rooms" if e.kind == "room" else "objects"]
        return f"dropped: {e.label} (confidence {e.conf:.2f} < threshold {t})", "unresolved"
    return "not linked to any detected element - possibly missed", "unresolved"


def find_blind_spots(els, hots, th):
    """What the answers cannot see: candidates below the thresholds, and uncertain symbol
    regions without any detection."""
    out = []
    for e in els.values():
        if e.status == "below_threshold" and e.kind not in ("text", "room"):
            t = th["objects"]
            out.append(BlindSpot(id=f"b-{e.id}", geometry=e.geometry, heuristic="below_threshold",
                                 reason=f"{e.label} candidate with confidence {e.conf:.2f} < threshold {t}: not in the graph",
                                 suggestion="check the image crop or lower the threshold"))
    for h in hots:
        if h.channel == "symbols" and not h.elements:
            out.append(BlindSpot(id=f"b-{h.id}", geometry=h.geometry, heuristic="uncertain_without_detection",
                                 reason="Layout model uncertain, but no symbol detected",
                                 suggestion="check the image crop: missed symbol?"))
    return out


# ============================================================================ answers

def _first_sentence(s: Optional[str], n=260):
    if not s:
        return ""
    m = re.split(r"(?<=[.!?])\s", s.strip(), maxsplit=1)
    return m[0][:n]


def _expand(refs, els, bg_tables):
    """Evidence entries -> element ids (+ non-element references). 'elements.doors' -> all its rows."""
    ids, other = [], []
    for r in refs or []:
        r = str(r)
        if r in els:
            ids.append(r)
        elif r in bg_tables:
            ids += [i for i in bg_tables[r] if i in els]
            other.append(r)
        else:
            other.append(r)
    return list(dict.fromkeys(ids)), list(dict.fromkeys(other))


def _ids_of_step(t, known):
    if t.get("ids"):
        return [i for i in t["ids"] if i in known]
    return [i for i in dict.fromkeys(ID_RE.findall(t.get("result") or "")) if i in known]


def _tool_title(t):
    a = t.get("args") or {}
    name = t["tool"]
    if name == "overview":
        return "Read the overview (tables, totals, notes)"
    if name == "query_table":
        where = a.get("where") or []
        where = where if isinstance(where, list) else [where]
        conds = " · ".join(f"{c.get('column')} {c.get('op', '=')} {c.get('value', '')}".strip() if isinstance(c, dict)
                           else str(c) for c in where)
        return f"Queried table {a.get('table')}" + (f" ({conds})" if conds else "")
    if name == "aggregate":
        return f"{a.get('op')} over {a.get('table')}" + (f".{a['column']}" if a.get("column") else "")
    if name == "search":
        return f"Searched for '{a.get('text')}'"
    if name == "get":
        return f"Read {a.get('path')}"
    if name == "calculate":
        return f"Calculated: {a.get('expression')}"
    if name == "view":
        return "Looked at the drawing" + (" (whole plan)" if a.get("id") == "plan" else " (crop)")
    if name == "final_answer":
        return "Gave the answer"
    return name


def _result_short(t):
    r = t.get("result") or ""
    m = re.search(r'"n_matched":\s*(\d+)', r)
    if m:
        return f"{m.group(1)} matches"
    m = re.search(r'"value":\s*([-\d.]+)', r)
    if m:
        return f"result {m.group(1)}"
    m = re.search(r'"n_hits":\s*(\d+)', r)
    if m:
        return f"{m.group(1)} hits"
    m = re.search(r'"result":\s*([-\d.]+)', r)
    if m:
        return f"= {m.group(1)}"
    if '"error"' in r:
        return "Error: " + r[:120]
    return None


def _region_of(t):
    m = re.search(r'"crop_px":\s*\[([^\]]+)\]', t.get("result") or "")
    if m:
        try:
            return [float(v) for v in m.group(1).split(",")][:4]
        except ValueError:
            return None
    b = (t.get("args") or {}).get("box_px")
    return [float(v) for v in b] if b else None


def build_answer(i, rec, els, bg_tables, th, blind) -> Answer:
    intent = rec.get("intent")
    kinds = INTENT_KINDS.get(intent, set())
    agent_used = "rule_prediction" in rec and not rec.get("llm_fallback")
    rule_val = rec.get("rule_prediction", rec.get("prediction"))
    rule_expl = rec.get("rule_explanation", rec.get("explanation"))
    rule_ev = rec.get("rule_evidence", rec.get("evidence"))
    ev_ids, refs = _expand(rec.get("evidence") if agent_used else rule_ev, els, bg_tables)
    if agent_used and not ev_ids:          # agent cited only paths ('facts_counts.doors'): use the rule evidence
        ev_ids, more = _expand(rule_ev, els, bg_tables)
        refs += more

    # evidence: counted / excluded (agent changes) / considered (candidates below threshold)
    excluded = {}
    for c in rec.get("llm_changes") or []:
        w = str(c.get("where", ""))
        if w in els and c.get("action") == "remove":
            excluded[w] = c.get("reason") or "removed by the agent"
        elif w in els and c.get("action") == "add" and w not in ev_ids:
            ev_ids.append(w)
    counted = [x for x in ev_ids if x not in excluded]
    counted.sort(key=lambda x: (els[x].geometry.bbox or [0, 0])[1::-1] if els[x].geometry.bbox else (0, 0))
    evidence = [Evidence(element=x, role="counted", order=n + 1,
                         reason=f"{els[x].label} (confidence {els[x].conf:.2f})" if els[x].conf is not None else els[x].label)
                for n, x in enumerate(counted)]
    evidence += [Evidence(element=x, role="excluded", reason=r) for x, r in excluded.items()]
    cands = [e for e in els.values() if e.status == "below_threshold" and e.kind in kinds]
    evidence += [Evidence(element=e.id, role="considered",
                          reason=f"below threshold {th['objects']} (confidence {e.conf:.2f}) - not counted") for e in cands]

    # sources, agreement, confidence
    sources = {"rules": SourceAnswer(value=rule_val, explanation=rule_expl)}
    if "rule_prediction" in rec:
        sources["agent"] = SourceAnswer(value=None if rec.get("llm_fallback") else rec.get("prediction"),
                                        text=rec.get("llm_answer"), explanation=rec.get("llm_reason"),
                                        model=(re.match(r"LLM \(([^)]+)\)", rec.get("explanation") or "") or [None, None])[1])
    vals = [s.value for s in sources.values() if s.value is not None]
    agreement = "single" if len(vals) < 2 else "agree" if len(set(vals)) == 1 else "conflict"
    conf = float(rec.get("confidence") or 0)
    label = "hoch" if conf >= 0.8 else "mittel" if conf >= 0.5 else "niedrig"

    # summary sentence
    noun = KIND_DE.get(next(iter(kinds)), "element") if kinds else "element"
    if agent_used and rec.get("llm_reason"):
        summary = _first_sentence(rec["llm_reason"])
    else:
        summary = _first_sentence(rule_expl)
    if cands:
        summary += f" {len(cands)} more {noun} candidate(s) were below the threshold {th['objects']} and were not counted."
    if excluded:
        summary += f" The agent removed {len(excluded)} element(s) from the rule count."

    # steps: routing, reading the plan, then the agent trace (or the rule)
    known = set(els)
    steps = [Step(n=1, kind="route", title=f"Question recognised as '{INTENT_DE.get(intent, intent or 'unknown')}'",
                  result="regex routing (qa.route)" if intent else "no known question type")]
    kept_k = [e for e in els.values() if e.status == "kept" and e.kind in kinds]
    if kinds:
        steps.append(Step(n=2, kind="read", title=f"Read the plan: {len(kept_k) + len(cands)} {noun} candidates detected",
                          result=f"{len(kept_k)} above threshold {th['objects']}, {len(cands)} below it dropped",
                          ids=[e.id for e in kept_k]))
    else:
        nr = sum(e.kind == "room" and e.status == "kept" for e in els.values())
        steps.append(Step(n=2, kind="read", title=f"Read the plan: {nr} rooms, "
                                                  f"{sum(e.kind == 'text' and e.status == 'kept' for e in els.values())} texts"))
    first_seen, views = {}, []
    if "rule_prediction" in rec and rec.get("trace"):
        for t in rec["trace"]:
            n = len(steps) + 1
            ids = _ids_of_step(t, known)
            if t["tool"] == "final_answer":
                ids = [x for x in (ids or counted) if x in known] or counted
                steps.append(Step(n=n, kind="final", tool="final_answer", title=f"Answer: {rec.get('prediction')}",
                                  result=_first_sentence((t.get('args') or {}).get('reason'), 200) or None, ids=ids))
                continue
            kind = "view" if t["tool"] == "view" else "read" if t["tool"] == "overview" else "tool"
            region = _region_of(t) if kind == "view" else None
            if region:
                views.append(region)
            steps.append(Step(n=n, kind=kind, tool=t["tool"], args=t.get("args") or None, title=_tool_title(t),
                              result=_result_short(t), ids=ids, region=region))
            for x in ids:
                first_seen.setdefault(x, n)
    else:
        steps.append(Step(n=len(steps) + 1, kind="final", title=f"Rule: {rule_val}", result=rule_expl, ids=counted))

    coverage = None
    if first_seen or views:
        by_kind = {}
        for k in sorted({e.kind for e in els.values() if e.status == "kept"}):
            tot = [e.id for e in els.values() if e.status == "kept" and e.kind == k]
            by_kind[k] = [sum(x in first_seen for x in tot), len(tot)]
        rel_blind = [b.id for b in blind if b.heuristic != "below_threshold"
                     or els.get(b.id[2:], Element(id="", kind="", geometry=Geometry(), status="kept")).kind in kinds]
        coverage = Coverage(by_kind=by_kind, first_seen=first_seen, views=views, blind_spots=rel_blind,
                            referenced_only=[x for x in dict.fromkeys(e.room for e in els.values()
                                                                      if e.id in first_seen and e.room)
                                             if x not in first_seen and x in els])
    limitations = []
    if intent == "showers_bathtubs":
        limitations.append("Showers have no symbol of their own in the layout model: one shower is assumed per bathroom without a tub.")
    if intent == "doors":
        limitations.append("The layout model does not recognise sliding doors and open passages as door symbols.")
    return Answer(id=f"q{i + 1}", question=rec["question"], intent=intent, value=rec.get("prediction"),
                  unit=rec.get("unit"), text=rec.get("llm_answer") or f"{rec.get('prediction')}",
                  summary=summary.strip(), confidence=round(conf, 2), confidence_label=label,
                  abstained=bool(rec.get("abstained")), agreement=agreement, sources=sources,
                  evidence=evidence, references=refs, steps=steps, coverage=coverage, limitations=limitations,
                  truth=rec.get("answer") if isinstance(rec.get("answer"), (int, float)) else None,
                  correct=rec.get("correct"))


# ============================================================================ main entry

def build_explanation(image: Image.Image, graph, det, probs, records, thresholds, models=None,
                      graph_file="plan_graph.json"):
    """-> (ExplanationDoc, {channel: RGBA heat image})."""
    from .build_graph import thresholds as norm_th
    from .plan_to_graph import to_building_graph

    th = norm_th(thresholds)
    els = build_elements(graph, det, probs, th)
    maps = uncertainty_maps(image.size, probs, det.get("words") if det else None)
    hots = find_hotspots(maps, probs, els, th)
    blind = find_blind_spots(els, hots, th)
    bg = json.loads(to_building_graph(graph).model_dump_json(exclude_none=True))
    bg_tables = {f"elements.{k}": [r[0] for r in v["rows"]] for k, v in (bg.get("elements") or {}).items()}
    bg_tables["spaces"] = [r[0] for r in (bg.get("spaces") or {}).get("rows", [])]
    answers = [build_answer(i, r, els, bg_tables, th, blind) for i, r in enumerate(records)]

    comb = cv2.blur(maps["combined"], (9, 9))
    unc = comb > 0.45
    share = {}
    if unc.any():
        chans = [c for c in ("rooms", "symbols", "text") if c in maps]
        arg = np.argmax(np.stack([maps[c] for c in chans]), 0)[unc]
        share = {c: round(float((arg == j).mean()), 3) for j, c in enumerate(chans)}
    doc = ExplanationDoc(
        source=Source(file=graph.source, kind="plan", page=graph.page, image_size=list(image.size), graph_file=graph_file),
        run=Run(created=datetime.now().astimezone().isoformat(timespec="seconds"), models=models or {}, thresholds=th),
        assets=Assets(plan_image=graph.source,
                      uncertainty_images={c: f"uncertainty_{c}.png" for c in maps},
                      uncertainty_share=share or None, uncertain_area=round(float(unc.mean()), 4)),
        elements=els, hotspots=hots, blind_spots=blind, answers=answers)
    return doc, {c: heat_rgba(m) for c, m in maps.items()}


def write_explanation(doc, heat: dict, image: Image.Image, out_dir) -> None:
    """explanation.json + uncertainty_<channel>.png + self-contained explain.html."""
    from pathlib import Path
    from .explain_view import explain_html
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "explanation.json").write_text(doc.model_dump_json(indent=1, exclude_none=True), encoding="utf-8")
    for c, im in heat.items():
        im.save(out_dir / f"uncertainty_{c}.png")
    explain_html(doc, image, heat, out_dir / "explain.html")
