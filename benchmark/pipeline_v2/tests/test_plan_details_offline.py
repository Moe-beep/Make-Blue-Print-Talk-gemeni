"""Offline test: what the plan path keeps beyond counts (no models, no network).

Wall runs from the Wall class, extra icon classes, rejected detections, room numbers, apartment
markers, plan scale and estimated areas, and that polygon / area_px reach building_graph.json.

    python tests/test_plan_details_offline.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from planqa.build_graph import apartment_key, build_graph
from planqa.building_schema import rows_of
from planqa.layout import objects_from_segmentation, walls_from_segmentation
from planqa.ocr import OcrWord
from planqa.pipeline import save_layout_maps
from planqa.plan_to_graph import to_building_graph
from planqa.schemas import GraphObject, GraphRoom, RoomType, TextKind

H, W = 400, 600
WALL, RAILING = 2, 8


def room(box, area_m2=None, conf=0.9, cls="Bedroom", rtype=RoomType.bedroom):
    x0, y0, x1, y1 = box
    return GraphRoom(
        id="",
        room_type=rtype,
        model_class=cls,
        type_conf=conf,
        type_reason=f"class {cls}",
        polygon_px=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        box_px=list(box),
        area_px=float((x1 - x0) * (y1 - y0)),
        area_m2=area_m2,
    )


def word(text, box, conf=0.95):
    return OcrWord(text, [float(v) for v in box], conf)


def test_walls_and_icons():
    prob = np.zeros((12, H, W), np.float32)
    prob[0] = 1.0
    for sl in [
        (slice(50, 60), slice(50, 550)),
        (slice(50, 350), slice(50, 60)),
        (slice(340, 350), slice(50, 550)),
    ]:
        prob[0][sl], prob[WALL][sl] = 0.0, 0.9
    prob[0][100:104, 300:500], prob[RAILING][100:104, 300:500] = 0.0, 0.8
    walls = walls_from_segmentation(prob)
    kinds = sorted((w.kind, w.orientation) for w in walls)
    assert kinds == [("railing", "h"), ("wall", "h"), ("wall", "h"), ("wall", "v")], (
        kinds
    )
    top = next(
        w
        for w in walls
        if w.kind == "wall" and w.orientation == "h" and w.box_px[1] == 50
    )
    assert top.length_px == 500 and abs(top.thickness_px - 10) < 0.5, top

    icons = np.zeros((11, H, W), np.float32)
    icons[0] = 1.0
    icons[0][10:30, 10:30], icons[4][10:30, 10:30] = 0, 0.9  # electrical appliance
    icons[0][10:30, 100:120], icons[10][10:30, 100:120] = 0, 0.9  # chimney
    assert sorted(o.kind for o in objects_from_segmentation(icons)) == [
        "chimney",
        "electrical_appliance",
    ]


def test_details_reach_building_graph():
    # 1 px = 0.02 m: 200x150 px = 12 m², 150x150 = 9 m², third room has no written area
    rooms = [
        room((0, 0, 200, 150), 12.0),
        room((200, 0, 350, 150), 9.0),
        room((0, 150, 200, 300)),
        room((350, 0, 500, 150), conf=0.5),
    ]  # below threshold
    words = [
        word("MH", [80, 40, 110, 55]),
        word("12,0 m2", [80, 60, 130, 75]),
        word("1.01", [80, 80, 110, 92]),
        word("MH", [260, 40, 290, 55]),
        word("9,0 m2", [260, 60, 300, 75]),
        word("1.02", [260, 80, 290, 92]),
        word("OH", [80, 200, 110, 215]),
        word("1.03", [80, 220, 110, 232]),
        word("A", [20, 20, 30, 35]),
        word("App. B", [20, 250, 70, 265]),
        word("2.40", [400, 300, 430, 312]),
        word("Pièce", [520, 300, 560, 315]),  # label outside every detected room
        word("1.07", [525, 318, 555, 330]),
        word("11,35 m2", [515, 334, 570, 348]),
        word("Bain", [10, 340, 40, 352]),  # two areas joined into one OCR line
        word("WC", [110, 340, 135, 352]),
        word("3.94 m² 4.04 m²", [0, 356, 150, 368]),
    ]
    objs = [
        GraphObject(
            id="", kind="door", box_px=[190, 50, 210, 90], area_px=400, conf=0.9
        ),
        GraphObject(
            id="", kind="door", box_px=[100, 140, 140, 160], area_px=400, conf=0.3
        ),
    ]
    pg = build_graph("plan.png", 0, (W, H), words, rooms, None, objs, [[0, 0, W, H]])
    kinds = {t.text: t.kind for t in pg.texts}
    assert (
        kinds["1.01"] == TextKind.room_number and kinds["2.40"] == TextKind.dimension
    ), kinds
    assert kinds["A"] == TextKind.apartment and kinds["App. B"] == TextKind.apartment, (
        kinds
    )
    assert (
        apartment_key("App. B") == "B"
        and apartment_key("a") is None
        and apartment_key("MH") is None
    )
    assert [r.id for r in pg.rejected_rooms] == ["rx01"] and [
        o.id for o in pg.rejected_objects
    ] == ["ox001"]
    assert pg.dropped == {"texts": 0, "rooms": 1, "objects": 1}
    assert (
        pg.scale
        and pg.scale["src"] == "room_areas"
        and abs(pg.scale["m_per_px"] - 0.02) < 1e-6
    ), pg.scale

    g = to_building_graph(pg)
    sp = {s["id"]: s for s in rows_of(g.spaces)}
    assert (
        sp["r01"]["number"] == "1.01"
        and sp["r02"]["number"] == "1.02"
        and sp["r03"]["number"] == "1.03"
    ), sp
    assert sp["r01"]["area_src"] == "label" and sp["r01"]["net_area"] == 12.0
    assert (
        sp["r03"]["area_src"] == "estimate:room_areas" and sp["r03"]["net_area"] == 12.0
    ), sp["r03"]
    assert (
        sp["r01"]["polygon_px"] == rooms[0].polygon_px
        and sp["r01"]["area_px"] == 30000.0
    )
    assert (
        sp["r01"]["model_type"] == "bedroom"
        and sp["r01"]["model_conf"] == 0.9
        and sp["r01"]["conf"] == 0.9
    )
    assert sp["r01"]["unit"] == "A" and sp["r03"]["unit"] == "B", (sp["r01"], sp["r03"])
    assert sp["r02"].get("unit") is None, sp[
        "r02"
    ]  # own region, no marker in it: unknown, not guessed
    # no marker inside any region -> nearest-marker fallback
    pg2 = build_graph(
        "plan.png",
        0,
        (W, H),
        [word("A", [5, 330, 15, 345])],
        [room((0, 0, 200, 150)), room((200, 0, 350, 150))],
        None,
        [],
        [[0, 0, W, H]],
    )
    sp2 = rows_of(to_building_graph(pg2).spaces)
    assert {s["unit"] for s in sp2} == {"A"} and all(
        "unit_from_nearest_marker" in s["issues"] for s in sp2
    ), sp2
    orphan = next(s for s in sp.values() if s["name"] == "Pièce")
    assert orphan["net_area"] == 11.35 and orphan["number"] == "1.07", orphan
    assert orphan["area_src"] == "label", orphan
    assert apartment_key("Asseu") is None and apartment_key("As. C") == "C"
    assert apartment_key("H") is None
    joined = {
        s["name"]: s.get("net_area") for s in sp.values() if s["name"] in ("Bain", "WC")
    }
    assert joined == {"Bain": 3.94, "WC": 4.04}, joined
    assert g.facts["counts"]["dwelling_units"] == 2 and set(
        g.facts["areas"]["by_unit"]
    ) == {"A", "B"}
    assert g.facts["counts"]["doors"] == 1  # the rejected door is not counted
    rej = {r["id"]: r for r in rows_of(g.elements["rejected"])}
    assert (
        rej["ox001"]["category"] == "door" and rej["rx01"]["category"] == "room:Bedroom"
    ), rej
    ann = {r["text"]: r["kind"] for r in rows_of(g.annotations)}
    assert ann["1.01"] == "room_number" and ann["A"] == "apartment"


def test_scale_from_dimensions_and_walls():
    prob = np.zeros((12, H, W), np.float32)
    prob[0] = 1.0
    for sl in [
        (slice(50, 60), slice(50, 550)),
        (slice(100, 300), slice(50, 60)),
        (slice(340, 350), slice(100, 400)),
    ]:
        prob[0][sl], prob[WALL][sl] = 0.0, 0.9
    walls = walls_from_segmentation(prob)
    # 500 px = 5000 mm, 200 px = 2,00 m, 300 px = 3000 mm  -> 0.01 m/px
    words = [
        word("5000", [280, 30, 320, 45]),
        word("2,00", [20, 190, 35, 210]),
        word("3000", [230, 355, 270, 370]),
    ]
    pg = build_graph("plan.png", 0, (W, H), words, [], None, [], [[0, 0, W, H]], walls)
    assert (
        pg.scale
        and pg.scale["src"] == "dimensions"
        and abs(pg.scale["m_per_px"] - 0.01) < 1e-6
    ), pg.scale
    g = to_building_graph(pg)
    wl = rows_of(g.elements["walls"])
    assert {w["length"] for w in wl} == {5.0, 2.0, 3.0}, wl
    assert g.storeys[0].walls["length"] == 10.0


def test_disagreeing_scale_is_not_used():
    rooms = [
        room((0, 0, 200, 150), 12.0),
        room((200, 0, 350, 150), 30.0),
        room((0, 150, 200, 300)),
    ]
    pg = build_graph("plan.png", 0, (W, H), [], rooms, None, [], [[0, 0, W, H]])
    assert pg.scale is None
    assert (
        next(s for s in rows_of(to_building_graph(pg).spaces) if s["id"] == "r03").get(
            "net_area"
        )
        is None
    )


def test_layout_maps_saved():
    room_prob, icon_prob, heat = (
        np.random.rand(n, 20, 30).astype(np.float32) for n in (12, 11, 21)
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "layout_maps.npz"
        save_layout_maps(path, room_prob, icon_prob, heat)
        with np.load(path) as z:
            assert set(z.files) == {"room_prob", "icon_prob", "heatmaps"} and z[
                "heatmaps"
            ].shape == (21, 20, 30)
            assert np.abs(z["room_prob"] / 255 - room_prob).max() <= 0.5 / 255 + 1e-6


if __name__ == "__main__":
    test_walls_and_icons()
    test_details_reach_building_graph()
    test_scale_from_dimensions_and_walls()
    test_disagreeing_scale_is_not_used()
    test_layout_maps_saved()
    print("plan details: all checks passed")
