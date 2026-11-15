"""house.json: everything known about one drawing in a single file.

{
  "_meta":  file(s), models, when, what each part is
  "answer": merged.campi.json - schema_campi.json with one value per field, the best model's
            (_meta.sources says which model, _meta.uncertain_fields where the models disagree)
  "models": {<model>: {"campi": <model>.campi.json, "building_graph": <model>.building_graph.json}}
            every model's complete output: its schema_campi facts and its building graph with all
            detections (rooms, symbols, walls, words, confidences, rejected detections)
}
"""

from __future__ import annotations

from datetime import UTC, datetime

HOUSE_SCHEMA = "house/1"
HOUSE_NAME = "house.json"
CONTENTS = {
    "answer": "schema_campi.json filled with one value per field, from the best model for that field "
    "(merged.campi.json)",
    "models": "complete output of every model: its schema_campi facts (campi) and its building graph "
    "(building_graph) with all detections",
}


def build_house(
    file: str, merged: dict, campi: dict[str, dict], graphs: dict[str, dict]
) -> dict:
    """campi / graphs: model name -> that model's <model>.campi.json / <model>.building_graph.json."""
    models = sorted(set(campi) | set(graphs))
    return {
        "_meta": {
            "schema": HOUSE_SCHEMA,
            "file": file,
            "models": models,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "contents": CONTENTS,
        },
        "answer": merged,
        "models": {
            m: {"campi": campi.get(m), "building_graph": graphs.get(m)} for m in models
        },
    }
