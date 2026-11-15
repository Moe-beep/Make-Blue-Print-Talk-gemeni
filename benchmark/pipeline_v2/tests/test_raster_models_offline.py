"""Offline test of the raster models: adapters to building graph, per-field merge, parallel jobs.

Fake model outputs, no torch. Run: python tests/test_raster_models_offline.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from pipelines.campi_fill import fill_campi
from pipelines.merge_campi import merge_campi
from pipelines.raster2seq_profiles import PROFILES
from pipelines.runners import (
    LAYOUT,
    MODEL_NAMES,
    PLAN2STRCT,
    RASTER2GRAPH,
    RASTER2SEQ,
    build_jobs,
)
from pipelines.schema_fill import load_schema
from pipelines.to_graph import RASTER2GRAPH_LABELS, raster2graph_graph, raster2seq_graph

SCHEMA = load_schema()
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def campi(graph, model: str) -> dict:
    return fill_campi(
        json.loads(graph.model_dump_json(exclude_none=True)), SCHEMA, model
    )


def test_raster2seq(work: Path, image: Path) -> None:
    out = work / RASTER2SEQ / "cubicasa5k" / "jsons"
    out.mkdir(parents=True)
    flat = [c for point in SQUARE for c in point]
    instances = [
        (3, SQUARE),
        (3, SQUARE),
        (1, SQUARE),
        (4, flat),
        (10, SQUARE),
        (10, SQUARE),
        (9, SQUARE),
    ]
    rows = [
        {"image_id": "plan", "segmentation": seg, "category_id": c, "id": i}
        for i, (c, seg) in enumerate(instances)
    ]
    (out / "plan.json").write_text(json.dumps(rows))

    fp = campi(
        raster2seq_graph(work / RASTER2SEQ, "plan", image, PROFILES["cubicasa"]),
        RASTER2SEQ,
    )["floor_plan"]
    assert fp["openings"] == {"window_count": 1, "door_count": 2}
    rc = fp["room_type_counts"]
    assert (rc["bedroom_count"], rc["kitchen_count"], rc["bathroom_count"]) == (2, 1, 1)
    assert (
        rc["wc_room_count"] is None and rc["sauna_count"] is None
    )  # no such CubiCasa class
    assert len(fp["detections"]["rooms"]) == 4 and len(fp["detections"]["doors"]) == 2
    # 256 canvas, 100x50 image: scale 2.56, top margin 64 -> canvas (0, 0) is pixel (0, -25)
    assert fp["detections"]["rooms"][0]["polygon_px"][:2] == [0.0, -25.0]

    # r2g512 checkpoint: no door/window classes -> unknown, not zero
    r2g = campi(
        raster2seq_graph(work / RASTER2SEQ, "plan", image, PROFILES["r2g512"]),
        RASTER2SEQ,
    )["floor_plan"]
    assert r2g["openings"] == {"window_count": None, "door_count": None}
    assert (
        raster2seq_graph(work / RASTER2SEQ, "missing", image, PROFILES["cubicasa"])
        is None
    )


def test_raster2graph(work: Path, image: Path) -> None:
    out = work / RASTER2GRAPH / "json"
    out.mkdir(parents=True)
    rooms = [
        {"semantic": s, "polygon": [[p[0] * 10, 128 + p[1] * 10] for p in SQUARE]}
        for s in ("bedroom", "restroom", "balcony")
    ]
    graph = {
        "image": "plan.png",
        "canvas_size": 512,
        "junctions": [[0, 128], [512, 128]],
        "wall_segments": [[[0, 128], [512, 128]]],
        "rooms": rooms,
    }
    (out / "plan.json").write_text(json.dumps(graph))

    fp = campi(
        raster2graph_graph(work / RASTER2GRAPH, "plan", image, RASTER2GRAPH_LABELS),
        RASTER2GRAPH,
    )["floor_plan"]
    rc = fp["room_type_counts"]
    assert (
        rc["bedroom_count"],
        rc["wc_room_count"],
        rc["balcony_count"],
        rc["kitchen_count"],
    ) == (1, 1, 1, 0)
    walls = fp["detections"]["walls"]
    assert (
        len(walls) == 1
        and walls[0]["length_px"] == 100.0
        and walls[0]["segment_px"] == [0.0, 0.0, 100.0, 0.0]
    )
    assert [j["point_px"] for j in fp["detections"]["wall_junctions"]] == [
        [0.0, 0.0],
        [100.0, 0.0],
    ]
    assert fp["openings"]["door_count"] is None


def per_model_doc(
    model: str, doors, windows, toilets=None, bedrooms=None, floors=None, door_list=()
) -> dict:
    facts = fill_campi({"source": {"kind": "image", "file": "p.png"}}, SCHEMA, model)
    fp = facts["floor_plan"]
    fp["openings"] = {"door_count": doors, "window_count": windows}
    fp["sanitary_fixtures"]["toilet_count"] = toilets
    fp["room_type_counts"]["bedroom_count"] = bedrooms
    fp["floor_count"] = floors
    fp["detections"]["doors"] = [{"confidence": c} for c in door_list]
    return facts


def test_merge() -> None:
    per_model = {
        PLAN2STRCT: per_model_doc(PLAN2STRCT, 9, 6, door_list=(0.9,) * 9),
        LAYOUT: per_model_doc(
            LAYOUT, 8, 6, toilets=2, bedrooms=3, floors=1, door_list=(0.8,) * 8
        ),
        RASTER2SEQ: per_model_doc(RASTER2SEQ, 11, 4, bedrooms=2),
    }
    merged = merge_campi(per_model, SCHEMA, "p.png")
    fp, meta = merged["floor_plan"], merged["_meta"]
    assert fp["openings"] == {"door_count": 9, "window_count": 6}
    assert meta["sources"]["floor_plan.openings.door_count"] == PLAN2STRCT
    assert (
        len(fp["detections"]["doors"]) == 9
    )  # the list comes from the model of the count
    assert (
        fp["sanitary_fixtures"]["toilet_count"] == 2 and fp["floor_count"] == 1
    )  # layout only
    assert fp["room_type_counts"]["bedroom_count"] == 3
    assert meta["sources"]["floor_plan.room_type_counts.bedroom_count"] == LAYOUT
    assert "floor_plan.openings.door_count" in meta["uncertain_fields"]  # 9 vs 8 vs 11
    assert (
        "floor_plan.openings.window_count" in meta["uncertain_fields"]
    )  # raster2seq says 4
    assert "floor_plan.floor_count" not in meta["uncertain_fields"]
    assert meta["values_by_model"]["floor_plan.openings.door_count"] == {
        PLAN2STRCT: 9,
        LAYOUT: 8,
        RASTER2SEQ: 11,
    }

    without_p2s = merge_campi(
        {k: v for k, v in per_model.items() if k != PLAN2STRCT}, SCHEMA, "p.png"
    )
    assert (
        without_p2s["floor_plan"]["openings"]["door_count"] == 8
    )  # next in the ranking: layout


def test_jobs(work: Path) -> None:
    jobs = build_jobs(work / "input", work, "cuda", MODEL_NAMES, PROFILES["r2g512"])
    assert [j.repo_dir.name for j in jobs] == [
        "Raster2Seq",
        "Raster-to-Graph",
        ROOT.name,  # project root (folder name does not matter)
        "2dplan2strct",
        "text_ocr",
        ROOT.name,
        ROOT.name,
    ]
    assert jobs[2].command[1:3] == ("-m", "pipelines.layout_infer")
    assert jobs[3].command[1].endswith("plan2strct_infer.py")
    assert jobs[4].command[1].endswith("text_ocr_infer.py")
    assert jobs[5].command[1].endswith("yolo_infer.py") and jobs[5].command[-2].endswith("floorplan-yolo11m-1280.onnx")
    assert jobs[6].command[-2].endswith("floorplan-symbols-yolo11l-1280.onnx")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        image = work / "plan.png"
        Image.new("RGB", (100, 50), "white").save(image)
        test_raster2seq(work, image)
        test_raster2graph(work, image)
        test_jobs(work)
    test_merge()
    print("OK")
