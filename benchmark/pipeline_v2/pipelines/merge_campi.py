"""Combine the per-model schema_campi files of one plan into merged.campi.json.

Field by field, the first model in that field's ranking (FIELD_RANKING, first matching pattern) that has a
value wins. A scalar on which a later-ranked model disagrees is listed in _meta.uncertain_fields. Maps and
lists (rooms, detections) are taken whole from one model, never mixed, so a list always matches the count
it explains (detections.doors comes from the same model as openings.door_count).
_meta.sources says which model each filled field came from.
"""

from __future__ import annotations

import copy
import fnmatch
from datetime import UTC, datetime

from pipelines.pdf_plan import PDF_PLAN
from pipelines.runners import (
    LAYOUT,
    PLAN2STRCT,
    RASTER2GRAPH,
    RASTER2SEQ,
    TEXT_OCR,
    YOLO_GENERAL,
    YOLO_SYMBOLS,
)
from pipelines.schema_fill import empty_facts

# The two YOLO detectors (models/yolo-weights) come first. Start: their app's class-by-class split, chosen on
# their validation set (symbols model for windows, sinks, showers/bathtubs; 13-class model for the rest).
# Then checked on our own validation split (work/val_input: 30 CubiCasa5K plans, none of the 30 test plans):
# the 13-class model counts windows (20/30 vs 17) and showers+bathtubs (26/27 vs 22) better, sinks are a tie
# (29/30 each) -> windows and showers/bathtubs moved to the 13-class model, sinks stay with the symbols model.
# The other models follow as fallback, in the order measured on 30 CubiCasa5K plans (exact counts):
#   doors 2DPlan2Strct 15/30, layout 11, Raster2Seq 6;  windows 13 / 10 / 4
#   toilets, sinks, floors: layout (24/30, 22/30, 30/30)
#   bedrooms layout 16/19, Raster2Seq 13;  kitchens 19/19 vs 16;  bathrooms 7/14 vs 6
#   room labels (YOLO26-Text + PaddleOCR): WC rooms 11/14, balconies 6/8, bathrooms 10/14, saunas 9/21
FALLBACK_OPENINGS = (PLAN2STRCT, LAYOUT, RASTER2SEQ)
FALLBACK_ROOMS = (LAYOUT, RASTER2SEQ, RASTER2GRAPH, PLAN2STRCT, TEXT_OCR)
FALLBACK_LABEL_ROOMS = (TEXT_OCR, LAYOUT, RASTER2SEQ, RASTER2GRAPH, PLAN2STRCT)
GENERAL_FIRST = (YOLO_GENERAL, YOLO_SYMBOLS)
SYMBOLS_FIRST = (YOLO_SYMBOLS, YOLO_GENERAL)

DOORS_RANKING = (*GENERAL_FIRST, *FALLBACK_OPENINGS)
WINDOWS_RANKING = (*GENERAL_FIRST, *FALLBACK_OPENINGS)
SINKS_RANKING = (*SYMBOLS_FIRST, LAYOUT)
SANITARY_RANKING = (*GENERAL_FIRST, LAYOUT)  # toilets, showers, bathtubs
ROOMS_RANKING = (PDF_PLAN, YOLO_GENERAL, *FALLBACK_ROOMS)
LABEL_ROOMS_RANKING = (PDF_PLAN, YOLO_GENERAL, *FALLBACK_LABEL_ROOMS)
FIELD_RANKING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("floor_plan.openings.door_count", DOORS_RANKING),
    ("floor_plan.openings.window_count", WINDOWS_RANKING),
    ("floor_plan.sanitary_fixtures.sink_and_washbasin_count", SINKS_RANKING),
    ("floor_plan.sanitary_fixtures.washbasin_count", SINKS_RANKING),
    ("floor_plan.sanitary_fixtures.*", SANITARY_RANKING),
    ("floor_plan.floor_count", (PDF_PLAN, YOLO_GENERAL, LAYOUT)),
    ("floor_plan.detections.doors", DOORS_RANKING),
    ("floor_plan.detections.windows", WINDOWS_RANKING),
    ("floor_plan.detections.sanitary", SANITARY_RANKING),
    ("floor_plan.detections.doorways", (YOLO_SYMBOLS,)),
    ("floor_plan.detections.rejected", DOORS_RANKING),
    ("floor_plan.detections.detection_threshold", DOORS_RANKING),
    ("floor_plan.detections.walls", (RASTER2GRAPH, PLAN2STRCT)),
    ("floor_plan.detections.wall_junctions", (RASTER2GRAPH,)),
    ("floor_plan.detections.storey_outlines", (YOLO_GENERAL, LAYOUT)),
    ("floor_plan.detections.rooms", ROOMS_RANKING),
    ("floor_plan.detections.texts", (TEXT_OCR,)),
    ("floor_plan.room_type_counts.sauna_count", LABEL_ROOMS_RANKING),
    ("floor_plan.room_type_counts.bathroom_count", LABEL_ROOMS_RANKING),
    ("floor_plan.room_type_counts.wc_room_count", LABEL_ROOMS_RANKING),
    ("floor_plan.room_type_counts.balcony_count", LABEL_ROOMS_RANKING),
    ("floor_plan.room_type_counts.*", ROOMS_RANKING),
    ("*", (PDF_PLAN, YOLO_GENERAL, YOLO_SYMBOLS, LAYOUT, PLAN2STRCT, RASTER2SEQ, RASTER2GRAPH, TEXT_OCR)),
)


# per-model evidence (thresholds, image size): differing values are expected, not a disagreement
NOT_COMPARED = "floor_plan.detections.*"


def ranking_for(path: str) -> tuple[str, ...]:
    return next(
        models
        for pattern, models in FIELD_RANKING
        if fnmatch.fnmatchcase(path, pattern)
    )


def is_empty(value) -> bool:
    return value is None or value == [] or value == {}


def leaf_paths(node: dict, path: str) -> list[tuple[str, bool]]:
    """(path, is_scalar) of every value merged as one unit: scalars, and whole maps / lists."""
    if node.get("type") == "object":
        return [
            lp
            for k, child in node.get("properties", {}).items()
            for lp in leaf_paths(child, f"{path}.{k}")
        ]
    return [(path, node.get("type") not in ("map", "list"))]


def get_path(doc: dict, path: str):
    for part in path.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None
    return doc


def set_path(doc: dict, path: str, value) -> None:
    *parents, last = path.split(".")
    for part in parents:
        doc = doc[part]
    doc[last] = value


def merge_campi(per_model: dict[str, dict], schema: dict, file: str) -> dict:
    """per_model: model name -> that model's campi document (fill_campi output)."""
    merged = empty_facts(schema)
    sources: dict[str, str] = {}
    uncertain: list[str] = []
    sections = sorted(
        {s for doc in per_model.values() for s in doc["_meta"]["sections"]}
    )
    for section in sections:
        for path, scalar in leaf_paths(schema[section], section):
            ranked = [
                (m, get_path(per_model[m], path))
                for m in ranking_for(path)
                if m in per_model
            ]
            available = [(m, v) for m, v in ranked if not is_empty(v)]
            if not available:
                continue
            model, value = available[0]
            set_path(merged, path, copy.deepcopy(value))
            sources[path] = model
            comparable = scalar and not fnmatch.fnmatchcase(path, NOT_COMPARED)
            if comparable and any(v != value for _, v in available[1:]):
                uncertain.append(path)
    meta = {
        "schema": "schema_campi.json",
        "rules": "planqa/campi_map.py",
        "file": file,
        "models": sorted(per_model),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": sources,
        "uncertain_fields": uncertain,
        "values_by_model": {
            path: {
                m: get_path(doc, path)
                for m, doc in per_model.items()
                if not is_empty(get_path(doc, path))
            }
            for path in uncertain
        },
    }
    return {"_meta": meta, **merged}
