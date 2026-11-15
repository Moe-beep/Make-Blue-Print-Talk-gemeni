"""Offline test of the common schema: plan graph -> building-graph, shared vocabulary, facts,
and the agent tools on both sources (no models, no network).

    python tests/test_building_graph_offline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_ifc_offline import model

from planqa.agent import Agent, JsonTools, ScriptedBackend
from planqa.building_schema import BuildingGraph, sanitary_category, space_categories
from planqa.ifc_extract import build_graph
from planqa.plan_to_graph import to_building_graph
from planqa.schemas import PlanGraph


def room(i, label, cls, box, area=None, conf=0.8):
    return {"id": f"r{i:02d}", "room_type": "other", "model_class": cls, "type_conf": conf, "type_reason": "",
            "polygon_px": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], "box_px": box,
            "area_px": float((box[2] - box[0]) * (box[3] - box[1])), "label_text": label, "area_m2": area}


def obj(i, kind, box, rid, conf=0.8):
    return {"id": f"o{i:02d}", "kind": kind, "box_px": box, "area_px": 100, "conf": conf, "room_id": rid}


def text(i, t, kind, box, rid=None, value=None):
    return {"id": f"t{i:02d}", "text": t, "kind": kind, "box_px": box, "conf": 0.95, "room_id": rid, "value": value,
            "unit": "m2" if value else None}


plan = PlanGraph.model_validate({
    "source": "plan.png", "page": 0, "image_size": [1000, 800],
    "rooms": [room(1, "OH+K", "Living Room", [0, 0, 400, 400], 30.5), room(2, "MH", "Bedroom", [400, 0, 700, 400], 12.0),
              room(3, "KH", "Bath", [700, 0, 1000, 200], 5.0), room(4, None, "Bath", [700, 200, 1000, 400], conf=0.7),
              room(5, None, "Outdoor", [0, 400, 300, 600]), room(6, "PARVEKE", "Outdoor", [300, 400, 600, 600], 6.0)],
    "texts": [text(1, "OH+K", "room_label", [10, 10, 60, 30], "r01"), text(2, "30,5 m2", "area", [10, 40, 60, 60], "r01", 30.5),
              text(3, "WC", "room_label", [800, 700, 830, 720]), text(4, "A", "apartment", [20, 300, 30, 315], "r01")],
    "objects": [obj(1, "door", [390, 100, 410, 160], "r01"), obj(2, "door", [690, 100, 710, 160], "r02"),
                obj(3, "window", [100, 0, 200, 10], "r01"), obj(4, "toilet", [900, 50, 950, 100], "r03"),
                obj(5, "sink", [720, 20, 760, 50], "r03"), obj(6, "sink", [300, 300, 350, 330], "r01"),
                obj(7, "bathtub", [720, 250, 900, 320], "r04"), obj(8, "closet", [420, 10, 500, 40], "r02")],
    "floors": [[0, 0, 1000, 800]], "min_conf": {"texts": 0.6, "rooms": 0.6, "objects": 0.6},
    "dropped": {"texts": 2, "rooms": 0, "objects": 1}})


def main():
    # ---- vocabulary
    assert space_categories("OH+K") == ["living", "kitchen"] and space_categories("KORIDOR (119)") == ["hallway"]
    assert space_categories("Wohnen/Essen") == ["living", "dining"] and space_categories("Meeting room") == ["meeting"]
    assert space_categories("Vano scala") == ["staircase"] and space_categories("Centre") == []
    assert sanitary_category("WASHHANDBASIN") == "washbasin" and sanitary_category("Badewanne") == "bathtub"

    # ---- plan -> building graph
    g = to_building_graph(plan)
    d = json.loads(g.model_dump_json(exclude_none=True))
    BuildingGraph.model_validate(d)
    assert d["schema_version"] == "building-graph/1" and d["source"]["kind"] == "image"
    assert d["source"]["extraction"]["dropped"]["texts"] == 2
    sp = {r[0]: dict(zip(d["spaces"]["columns"], r)) for r in d["spaces"]["rows"]}
    assert sp["r01"]["categories"] == ["living", "kitchen"] and sp["r01"]["src"] == "room_label"
    assert sp["r04"]["categories"] == ["bathroom"] and sp["r04"]["src"] == "room_polygon" and sp["r04"]["conf"] == 0.7
    assert "categories" not in sp["r05"] or not sp["r05"]["categories"], sp["r05"]     # unlabelled outdoor, plan names balconies
    assert sp["t03"]["categories"] == ["wc"] and "label_without_detected_room" in sp["t03"]["issues"]
    san = {r[0]: dict(zip(d["elements"]["sanitary"]["columns"], r)) for r in d["elements"]["sanitary"]["rows"]}
    assert san["o05"]["category"] == "washbasin" and san["o06"]["category"] == "sink"
    assert san["r03.shower"]["src"] == "inferred" and "r04.shower" not in san            # r04 has a bathtub
    f = d["facts"]["counts"]
    assert (f["doors"], f["windows"], f["toilets"], f["sinks_and_washbasins"], f["showers_and_bathtubs"]) == (2, 1, 1, 2, 2), f
    assert (f["kitchens"], f["living_rooms"], f["bedrooms"], f["bathrooms"], f["wc_rooms"], f["balconies"]) == (1, 1, 1, 2, 1, 1), f
    a = d["facts"]["areas"]
    assert a["net_floor_area"] == 47.5 and a["outdoor_area"] == 6.0 and a["by_name"]["MH"]["net_area"] == 12.0, a
    assert d["storeys"][0]["counts"]["doors"] == 2 and d["storeys"][0]["net_floor_area"] == 47.5

    # ---- the same tools and the same paths on both sources
    ifc = json.loads(build_graph(model()).model_dump_json(exclude_none=True))
    for doc, kind in ((plan.model_dump(mode="json"), "plan"), (d, "plan"), (ifc, "ifc")):
        T = JsonTools(doc)
        assert T.kind == kind and {"spaces", "elements.doors", "elements.sanitary"} <= set(T.tables), (kind, T.tables)
        ov = T.overview()
        assert ov["facts_counts"]["doors"] == 2 and "bathrooms" in ov["facts_counts"]
        n = T.query_table("spaces", where=[{"column": "categories", "op": "contains", "value": "bathroom"}])["n_matched"]
        assert n == ov["facts_counts"]["bathrooms"], (kind, n)
        assert T.aggregate("elements.sanitary", "count", group_by="category")["groups"]["toilet"] == 1
    # plan_graph.json given directly is converted transparently
    r = Agent(ScriptedBackend([[("get", {"path": "facts.counts.doors"})],
                               [("final_answer", {"answer": "2", "value": 2, "evidence": ["o01", "o02"],
                                                  "confidence": 0.8, "reason": "facts"})]])).ask(plan.model_dump(mode="json"), "doors?")
    assert r.value == 2 and r.trace[1]["result"] == '{"result": 2}', r.trace
    # ---- label robustness: OCR typos, labels from other texts, merged rooms, split open plans, opening marks
    p2 = PlanGraph.model_validate({
        "source": "p2.png", "page": 0, "image_size": [600, 400],
        "rooms": [room(1, "PARVEXE", "Outdoor", [0, 0, 200, 100]), room(2, None, "Outdoor", [300, 0, 500, 100]),
                  room(3, "AVAVATIN", "Living Room", [0, 120, 200, 300]), room(4, None, "Kitchen", [205, 200, 300, 300]),
                  room(5, "KH", "Bath", [400, 120, 500, 200]), room(6, None, "Bedroom", [300, 300, 400, 400]),
                  room(7, None, "Bath", [505, 120, 600, 200])],
        "texts": [text(1, "PARVEXE", "room_label", [10, 10, 60, 20], "r01"), text(2, "OH+RT+K", "other", [10, 200, 60, 210], "r03"),
                  text(3, "S", "room_label", [450, 150, 460, 160], "r05"), text(4, "MH4 (VARAUS)", "other", [310, 310, 380, 320], "r06"),
                  text(5, "9x21", "other", [100, 330, 120, 340]), text(6, "M.AKUUHVONE", "other", [520, 150, 590, 160], "r07")],
        "objects": [], "floors": [], "min_conf": 0.6, "dropped": {}})
    d2 = json.loads(to_building_graph(p2).model_dump_json(exclude_none=True))
    s2 = {r[0]: dict(zip(d2["spaces"]["columns"], r)) for r in d2["spaces"]["rows"]}
    assert s2["r01"]["categories"] == ["balcony"] and s2["r01"]["src"] == "room_label", s2["r01"]   # PARV* abbreviation
    assert s2["r03"]["categories"] == ["living", "kitchen"] and s2["r03"]["src"] == "room_text"   # from 'OH+RT+K'
    assert s2["r04"]["categories"] == ["kitchen"] and s2["r04"]["src"] == "room_polygon"   # no merging heuristics
    assert s2["t03"]["categories"] == ["sauna"] and "second_label_in_room:r05" in s2["t03"]["issues"]
    assert s2["r06"]["categories"] == ["bedroom"] and "reserved_room_VARAUS" in s2["r06"]["issues"]
    assert s2["r07"]["categories"] == ["bedroom"] and "ocr_fuzzy:MAKUUHUONE" in s2["r07"]["issues"]   # beats model 'Bath'
    c2 = d2["facts"]["counts"]
    assert (c2["balconies"], c2["kitchens"], c2["saunas"], c2["bedrooms"], c2["bathrooms"]) == (1, 2, 1, 2, 1), c2
    kinds = {r[0]: r[2] for r in d2["annotations"]["rows"]}
    assert kinds["t05"] == "opening_size" and kinds["t02"] == "other"
    from planqa.plan_to_graph import room_answer
    ra = room_answer(to_building_graph(p2), "kitchens")
    assert ra.value == 2 and ra.evidence == ["r03", "r04"], ra
    rb = room_answer(to_building_graph(p2), "balconies")
    assert rb.value == 1 and "unlabelled_outdoor" in rb.explanation, rb

    # ---- view tool: crops of the drawing, image passed to the backend, not into the trace
    from PIL import Image
    img = Image.new("RGB", (600, 400), "white")
    T = JsonTools(d2, image=img)
    v = T.view(id="r05")
    assert v["shown"] == "r05" and v["_image"][:4] == b"\x89PNG" and v["crop_px"][0] < 400, v["crop_px"]
    assert T.view(id="plan")["shown"] == "whole drawing" and T.view(box_px=[0, 0, 50, 50])["views_left"] == 0
    try:
        T.view(id="r01"); raise AssertionError("view limit")
    except ValueError:
        pass
    be = ScriptedBackend([[("view", {"id": "r07"})],
                          [("final_answer", {"answer": "2", "value": 2, "evidence": ["r06", "r07", "view:r07"],
                                             "confidence": 0.8, "reason": "label read in the view"})]])
    r = Agent(be).ask(d2, "bedrooms?", image=img)
    assert r.value == 2 and r.trace[1]["tool"] == "view" and "_image" not in r.trace[1]["result"], r.trace
    assert "_image" in be.seen[0][0][1]                                # the backend received the PNG
    r = Agent(ScriptedBackend([[("view", {"id": "r07"})], [("final_answer", {"answer": "x", "confidence": 0.1,
                                                                            "reason": "-"})]])).ask(d2, "q")
    assert "error" in json.loads(r.trace[1]["result"])                 # no drawing given -> no view tool

    # ---- overview vs. detail resolution
    big = Image.new("RGB", (3000, 2000), "white")
    T = JsonTools(d2, image=big)
    ov = T.view(id="plan")
    assert ov["resolution"] == "overview" and ov["scale"] < 1 and "NOT for counting" in ov["note"], ov
    det = T.view(id="r05")
    assert det["resolution"] == "full" and det["scale"] >= 1, det
    assert T.view(box_px=[0, 0, 2000, 1500])["resolution"] == "reduced"

    # ---- a count that differs from the estimate must be backed item by item
    def run(changes, views=(), est=1, value=2):
        steps = [[("view", {"id": v})] for v in views]
        steps.append([("final_answer", {"answer": str(value), "value": value, "confidence": 0.9, "reason": "seen",
                                        "changes": changes})])
        return Agent(ScriptedBackend(steps)).ask(d2, "balconies?", image=img, estimate=est)
    r = run([])
    assert r.value == 1 and r.override.startswith("rejected") and "no changes" in r.override, r.override
    r = run([{"action": "add", "item": "balcony", "where": "view:plan"}], views=["plan"])
    assert r.value == 1 and "neither a data row nor a detailed view" in r.override, r.override
    r = run([{"action": "add", "item": "balcony", "where": "r02 view:r02", "reason": "mirrored balcony"}], views=["r02"])
    assert r.value == 2 and r.override == "accepted", r.override
    r = run([{"action": "add", "item": "balcony", "where": "view:r02"}], views=["r02"], value=3)
    assert r.value == 1 and "add up to +1" in r.override, r.override
    r = run([], value=1)
    assert r.value == 1 and r.override is None                       # same as the estimate: nothing to check

    print(f"building graph: all checks passed (plan {len(json.dumps(d)) / 1024:.1f} KB, ifc {len(json.dumps(ifc)) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
