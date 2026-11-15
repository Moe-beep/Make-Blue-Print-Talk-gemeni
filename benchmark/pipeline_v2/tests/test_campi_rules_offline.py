"""Offline test: model output -> building graph -> rules (planqa/campi_map.py) -> schema_campi.

Fake model outputs, no torch. Run: python tests/test_campi_rules_offline.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from pipelines.campi_fill import fill_campi
from pipelines.house import build_house
from pipelines.merge_campi import merge_campi
from pipelines.schema_fill import load_schema
from pipelines.to_graph import (
    layout_graph,
    letterbox_to_image,
    pdf_plan_graph,
    plan2strct_graph,
    text_ocr_graph,
    yolo_graph,
)

SCHEMA = load_schema()


def as_doc(graph) -> dict:
    return json.loads(graph.model_dump_json(exclude_none=True))


def test_letterbox() -> None:
    # 200x100 image on a 512 canvas: scale 2.56, centred vertically (top margin 128)
    assert letterbox_to_image([[0, 128], [512, 384]], 512, 200, 100) == [
        [0.0, 0.0],
        [200.0, 100.0],
    ]


def test_plan2strct(work: Path, image: Path) -> None:
    out = work / "plan2strct"
    out.mkdir()
    detections = [
        {"label": "door", "score": 0.9, "box_xyxy": [0, 0, 5, 5]},
        {
            "label": "door",
            "score": 0.3,
            "box_xyxy": [9, 9, 12, 12],
        },  # below 0.35: kept as rejected
        {"label": "window", "score": 0.8, "box_xyxy": [20, 0, 30, 2]},
        {"label": "wall", "score": 0.7, "box_xyxy": [0, 0, 100, 3]},
        {"label": "room", "score": 0.95, "box_xyxy": [0, 0, 50, 40]},
    ]
    (out / "plan.json").write_text(
        json.dumps(
            {
                "image": "plan.png",
                "width": 100,
                "height": 50,
                "min_score": 0.2,
                "detections": detections,
            }
        )
    )
    doc = as_doc(plan2strct_graph(out, "plan", image))
    assert (
        len(doc["elements"]["rejected"]["rows"]) == 1
        and len(doc["elements"]["walls"]["rows"]) == 1
    )
    campi = fill_campi(doc, SCHEMA, "plan2strct")
    fp = campi["floor_plan"]
    assert fp["openings"] == {"window_count": 1, "door_count": 1}
    # no room types, no sanitary classes, no storey separation: unknown, not zero
    assert (
        fp["room_type_counts"]["bedroom_count"] is None
        and fp["sanitary_fixtures"]["toilet_count"] is None
    )
    assert fp["floor_count"] is None
    assert (
        campi["_meta"]["filled"]["floor_plan.openings.door_count"]
        == "facts.counts.doors"
    )
    assert (
        campi["bim_model"]["element_counts"]["door"] is None
    )  # an image fills floor_plan only


def test_layout(work: Path, image: Path) -> None:
    out = work / "layout"
    out.mkdir()
    objects = [
        {"kind": k, "conf": c, "box_px": [0, 0, 4, 4], "area_px": 16}
        for k, c in (
            ("door", 0.9),
            ("window", 0.7),
            ("toilet", 0.8),
            ("sink", 0.9),
            ("door", 0.4),
        )
    ]
    rooms = [
        {
            "room_type": "bedroom",
            "model_class": "Bedroom",
            "type_conf": 0.9,
            "fixtures": [],
            "polygon_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "box_px": [0, 0, 10, 10],
            "area_px": 100.0,
        }
    ]
    (out / "plan.json").write_text(
        json.dumps(
            {
                "image": "plan.png",
                "image_size_px": [100, 50],
                "objects": objects,
                "rooms": rooms,
                "floors": [[0, 0, 100, 50]],
            }
        )
    )
    campi = fill_campi(as_doc(layout_graph(out, "plan", image)), SCHEMA, "layout")
    fp = campi["floor_plan"]
    assert fp["openings"] == {"window_count": 1, "door_count": 1}
    assert (
        fp["sanitary_fixtures"]["toilet_count"] == 1
        and fp["sanitary_fixtures"]["sink_and_washbasin_count"] == 1
    )
    assert (
        fp["sanitary_fixtures"]["shower_count"] is None
        and fp["sanitary_fixtures"]["washbasin_count"] is None
    )
    assert (
        fp["room_type_counts"]["bedroom_count"] == 1
        and fp["room_type_counts"]["kitchen_count"] == 0
    )
    assert fp["floor_count"] == 1
    assert "showers" in campi["_meta"]["not_measured_by_model"]


def test_pdf_plan() -> None:
    rooms = {
        "5.01": {
            "label": "Balcone",
            "net_floor_area_m2": 5.37,
            "area_m2_precise": 5.3701,
            "apartment": "A",
            "excluded_from_net_area": "outdoor",
        },
        "5.02": {
            "label": "Stanza",
            "net_floor_area_m2": 15.84,
            "area_m2_precise": 15.8449,
            "apartment": "A",
            "excluded_from_net_area": None,
        },
        "5.03": {
            "label": "Stanza",
            "net_floor_area_m2": 10.15,
            "area_m2_precise": 10.1549,
            "apartment": "A",
            "excluded_from_net_area": None,
        },
        "5.04": {
            "label": None,
            "net_floor_area_m2": 1.5,
            "area_m2_precise": 1.5,
            "apartment": None,
            "excluded_from_net_area": "shaft_or_void",
        },
    }
    plan = {
        "scale_denominator": 100,
        "rooms": rooms,
        "apartments": {"A": {"net_floor_area_m2": 26.0}},
        "total_net_floor_area_m2": 26.0,
    }
    campi = fill_campi(
        as_doc(pdf_plan_graph(plan, Path("plan.pdf"))), SCHEMA, "pdf_plan"
    )
    fp = campi["floor_plan"]
    # sums of the precise areas, rounded once: 15.8449 + 10.1549 = 25.9998 -> 26.0 (not 15.84 + 10.15 = 25.99)
    assert fp["total_net_floor_area_m2"] == 26.0 and fp["apartments"] == {
        "A": {"net_floor_area_m2": 26.0}
    }
    assert fp["rooms"]["5.03"] == {"label": "Stanza", "net_floor_area_m2": 10.15}
    assert fp["rooms_by_label"]["Stanza"] == {
        "room_count": 2,
        "total_net_area_m2": 26.0,
    }
    assert fp["apartment_count"] == 1 and fp["room_type_counts"]["balcony_count"] == 1
    assert fp["rooms"]["5.04"]["net_floor_area_m2"] is None  # shaft: no net area


def test_text_ocr(work: Path, image: Path) -> None:
    out = work / "text_ocr"
    out.mkdir()
    square = lambda x, y: [[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]]
    words = [("MH", 0.95), ("MH", 0.9), ("OH+K", 0.9), ("KH", 0.8), ("WC", 0.7), ("S", 0.9),
             ("PARVEKE", 0.85), ("M.AKUUHVONE", 0.9), ("10,5 m2", 0.9), ("WC", 0.4)]  # last: too unsure
    texts = [{"text": w, "confidence": c, "detection_confidence": 0.8, "line": i, "polygon": square(i * 10, 0)}
             for i, (w, c) in enumerate(words)]
    (out / "plan.json").write_text(json.dumps({"image": "plan.png", "width": 100, "height": 50,
                                               "model": "yolo26x-text + paddleocr", "texts": texts}))
    campi = fill_campi(as_doc(text_ocr_graph(out, "plan", image)), SCHEMA, "text_ocr")
    rc = campi["floor_plan"]["room_type_counts"]
    # MH x2 + misspelt MAKUUHUONE = 3 bedrooms; OH+K counts as a kitchen; the 0.4 WC is not counted
    assert (rc["bedroom_count"], rc["kitchen_count"], rc["bathroom_count"]) == (3, 1, 1), rc
    assert (rc["wc_room_count"], rc["sauna_count"], rc["balcony_count"]) == (1, 1, 1), rc
    assert campi["floor_plan"]["openings"]["door_count"] is None  # words only
    det = campi["floor_plan"]["detections"]
    assert len(det["texts"]) == len(words) and len(det["rooms"]) == 8
    assert [t["kind"] for t in det["texts"]][-2:] == ["text", "text"]
    assert det["texts"][0]["box_px"] == [0.0, 0.0, 10.0, 5.0]

    # merged: the room labels count WC rooms and saunas first, the layout model keeps bedrooms and kitchens
    layout = fill_campi(as_doc(layout_graph(work / "layout", "plan", image)), SCHEMA, "layout")
    merged = merge_campi({"layout": layout, "text_ocr": campi}, SCHEMA, "plan.png")
    sources = merged["_meta"]["sources"]
    assert merged["floor_plan"]["room_type_counts"]["wc_room_count"] == 1
    assert sources["floor_plan.room_type_counts.wc_room_count"] == "text_ocr"
    assert sources["floor_plan.room_type_counts.sauna_count"] == "text_ocr"
    assert sources["floor_plan.room_type_counts.bedroom_count"] == "layout"
    assert "floor_plan.room_type_counts.bedroom_count" in merged["_meta"]["uncertain_fields"]  # layout 1, OCR 3


def test_yolo(work: Path, image: Path) -> None:
    out = work / "yolo"
    out.mkdir()
    spec = {"classes": ["window", "door", "toilet", "sink", "shower", "bathtub", "doorway", "sauna", "wc_room",
                        "floor"],
            "thresholds": {"window": 0.5, "door": 0.5, "toilet": 0.5, "sink": 0.5, "shower": 0.3, "bathtub": 0.5,
                           "doorway": 0.4, "sauna": 0.5, "wc_room": 0.2, "floor": 0.3},
            "thresholds_onnx_path": {"window": 0.5, "door": 0.6, "toilet": 0.5, "sink": 0.5, "shower": 0.3,
                                     "bathtub": 0.5, "doorway": 0.4, "sauna": 0.5, "wc_room": 0.2, "floor": 0.3}}
    found = [("door", 0.9), ("door", 0.55), ("window", 0.7), ("doorway", 0.8), ("shower", 0.35), ("bathtub", 0.4),
             ("toilet", 0.9), ("wc_room", 0.25), ("floor", 0.9)]
    detections = [{"label": k, "score": c, "box_xyxy": [0, 0, 10, 10]} for k, c in found]
    (out / "plan.json").write_text(json.dumps({"image": "plan.png", "width": 100, "height": 50, "model": "m.onnx",
                                               "detections": detections}))
    fp = fill_campi(as_doc(yolo_graph(out, "plan", image, spec, "yolo_general")), SCHEMA, "yolo_general")["floor_plan"]
    # ONNX-path thresholds: the 0.55 door is below 0.6; the doorway is not a door
    assert fp["openings"] == {"window_count": 1, "door_count": 1}
    sf = fp["sanitary_fixtures"]
    assert (sf["shower_count"], sf["bathtub_count"], sf["shower_and_bathtub_count"], sf["toilet_count"]) == (1, 0, 1, 1)
    assert sf["sink_and_washbasin_count"] == 0  # sink class known, none found
    rc = fp["room_type_counts"]
    assert (rc["wc_room_count"], rc["sauna_count"], rc["kitchen_count"]) == (1, 0, None)  # no kitchen class
    assert fp["floor_count"] == 1
    assert len(fp["detections"]["doorways"]) == 1 and len(fp["detections"]["rejected"]) == 2


def test_house(work: Path, image: Path) -> None:
    graph = as_doc(layout_graph(work / "layout", "plan", image))
    campi = fill_campi(graph, SCHEMA, "layout")
    merged = merge_campi({"layout": campi}, SCHEMA, "plan.png")
    house = build_house("plan.png", merged, {"layout": campi}, {"layout": graph})
    assert house["_meta"]["models"] == ["layout"] and house["answer"] is merged
    assert house["models"]["layout"]["campi"] is campi and house["models"]["layout"]["building_graph"] is graph
    assert house["answer"]["floor_plan"]["openings"]["door_count"] == 1


if __name__ == "__main__":
    test_letterbox()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        image = work / "plan.png"
        Image.new("RGB", (100, 50), "white").save(image)
        test_plan2strct(work, image)
        test_layout(work, image)
        test_text_ocr(work, image)
        test_yolo(work, image)
        test_house(work, image)
    test_pdf_plan()
    print("OK")
