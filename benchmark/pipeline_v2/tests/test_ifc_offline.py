"""Offline test of the IFC builder with a synthetic raw model (no ifcopenshell needed).

    python tests/test_ifc_offline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planqa.ifc_extract import build_graph
from planqa.ifc_schema import IfcGraph


def rect(x0, y0, x1, y1, z0=0.0, z1=3.0):
    a, b, c, d = [x0, y0], [x1, y0], [x1, y1], [x0, y1]
    return {"tris": [[a, b, c], [a, c, d]], "zmin": z0, "zmax": z1}


N = [0]


def el(cls, storey, name=None, psets=None, type_=None, **kw):
    N[0] += 1
    e = {"eid": 1000 + N[0], "guid": f"G{N[0]:04d}", "cls": cls, "name": name, "long_name": None, "object_type": None,
         "predefined": None, "is_mep": False, "type": type_, "storey_eid": storey, "space_eid": None, "host_eid": None,
         "psets": psets or {}, "material": {}, "pos": None, "attrs": {}, "profile": None, "flights": []}
    e.update(kw)
    return e


def model():
    G, U = 1, 2
    wt = {"eid": 501, "cls": "IfcWallType", "name": "AW 36.5 Ziegel", "predefined": "SOLIDWALL"}
    it = {"eid": 502, "cls": "IfcWallType", "name": "IW 11.5", "predefined": "PARTITIONING"}
    ext = {"Pset_WallCommon": {"IsExternal": True, "LoadBearing": True, "FireRating": "REI90"}}
    inn = {"Pset_WallCommon": {"IsExternal": False, "LoadBearing": False}}
    lay = {"materials": ["Putz", "Ziegel"], "layers": [["Putz", 0.015], ["Ziegel", 0.35]]}
    E, S = [], {}

    def add(e, shape=None):
        E.append(e)
        if shape:
            S[e["eid"]] = shape
        return e

    wohnen = add(el("IfcSpace", G, "001", {"Qto_SpaceBaseQuantities": {"NetFloorArea": 20.0, "Height": 2.6}},
                    long_name="Wohnen"), rect(0, 0, 5, 4, 0, 2.6))
    bad = add(el("IfcSpace", G, "002", {"Qto_SpaceBaseQuantities": {"NetFloorArea": 11.2}}, long_name="Bad"),
              rect(5.2, 0, 8, 4, 0, 2.5))
    add(el("IfcSpace", U, "101", {"Qto_SpaceBaseQuantities": {"NetFloorArea": 32.0}}, long_name="Schlafen"),
        rect(0, 0, 8, 4, 3, 5.6))
    walls = []
    for x0, y0, x1, y1, q in [(-0.365, -0.365, 8.365, 0, 8.73), (-0.365, 4, 8.365, 4.365, 8.73),
                              (-0.365, 0, 0, 4, 4.0), (8, 0, 8.365, 4, 4.0)]:
        walls.append(add(el("IfcWall", G, "AW", {**ext, "Qto_WallBaseQuantities": {"Length": q, "Height": 3.0, "NetSideArea": q * 3 - 2,
                                                                                      "NetVolume": q * 3 * 0.365}},
                            type_=wt, material=lay), rect(x0, y0, x1, y1)))
    iw = add(el("IfcWall", G, "IW", {**inn, "Qto_WallBaseQuantities": {"Length": 4.0, "NetSideArea": 10.0, "NetVolume": 1.2}},
                type_=it, material={"materials": ["Gipskarton"]}), rect(5, 0, 5.2, 4))
    add(el("IfcWall", U, "AW", {**ext, "Qto_WallBaseQuantities": {"Length": 24.0}}, type_=wt, material=lay), rect(-0.365, -0.365, 8.365, 0, 3, 6))
    add(el("IfcDoor", G, "T1", {"Pset_DoorCommon": {"IsExternal": False}}, host_eid=iw["eid"],
           attrs={"OverallWidth": 0.885, "OverallHeight": 2.01, "OperationType": "SINGLE_SWING_LEFT"}), rect(5, 1.5, 5.2, 2.4))
    add(el("IfcDoor", G, "HT", {"Pset_DoorCommon": {"IsExternal": True, "FireRating": "EI30"}}, host_eid=walls[0]["eid"],
           attrs={"OverallWidth": 1.01, "OverallHeight": 2.135}), rect(1, -0.365, 2, 0))
    add(el("IfcWindow", G, "F1", {"Pset_WindowCommon": {"IsExternal": True}}, host_eid=walls[1]["eid"],
           attrs={"OverallWidth": 1.5, "OverallHeight": 1.2}), rect(2, 4, 3.5, 4.365, 0.9, 2.1))
    add(el("IfcWindow", G, "F2", {"Pset_WindowCommon": {"IsExternal": True}}, host_eid=walls[3]["eid"],
           attrs={"OverallWidth": 0.8, "OverallHeight": 0.8}), rect(8, 1, 8.365, 1.8, 1.4, 2.2))
    st = add(el("IfcStair", G, "Treppe", {"Pset_StairCommon": {"NumberOfRiser": 17}},
                flights=[{"risers": 17, "riser_height": 0.176, "tread_length": 0.27}]), rect(0.5, 0.5, 1.5, 3.5, 0, 3))
    add(el("IfcSlab", G, "Bodenplatte", {"Qto_SlabBaseQuantities": {"NetArea": 38.0, "NetVolume": 9.5, "Depth": 0.25}},
           predefined="BASESLAB"))
    add(el("IfcSlab", U, "Decke", {"Qto_SlabBaseQuantities": {"NetArea": 36.0, "NetVolume": 7.2}}, predefined="FLOOR"))
    add(el("IfcColumn", G, "Stütze", profile={"extrusion": 3.0, "shape": "rectangle", "x": 0.3, "y": 0.3},
           psets={"Pset_ColumnCommon": {"LoadBearing": True}}))
    add(el("IfcSanitaryTerminal", G, "WC", predefined="TOILETPAN", pos=[6.0, 3.5, 0.0]))
    add(el("IfcSanitaryTerminal", G, "Waschtisch", predefined="WASHHANDBASIN", space_eid=bad["eid"]))
    for _ in range(2):
        add(el("IfcFurniture", G, "Stuhl:Stuhl:123", pos=[2.0, 2.0, 0.0]))
    pipes = [add(el("IfcPipeSegment", G, "Rohr", {"Qto_PipeSegmentBaseQuantities": {"Length": 3.5}}, is_mep=True)) for _ in range(3)]
    return {"header": {"file": "synthetic.ifc", "schema": "IFC4", "application": "TestCAD 1.0", "view_definition": "ViewDefinition [ReferenceView]"},
            "building": {"project": "Testprojekt", "site": "Grundstück", "name": "Haus A", "address": "Hauptstr. 1, Wien",
                         "lat": 48.2, "lon": 16.37, "georef": False, "true_north_deg": 0.0},
            "storeys": [{"eid": U, "name": "OG", "elevation": 3.0, "psets": {}}, {"eid": G, "name": "EG", "elevation": 0.0, "psets": {}}],
            "elements": E, "space_boundaries": [],
            "systems": [{"name": "Trinkwasser", "kind": "DOMESTICCOLDWATER", "members": [p["eid"] for p in pipes]}],
            "shapes": S, "_ids": {"wohnen": wohnen, "bad": bad, "stair": st}}


def main():
    raw = model()
    g = build_graph(raw)
    d = json.loads(g.model_dump_json(exclude_none=True))
    IfcGraph.model_validate(d)
    by = {x["name"]: x for x in d["storeys"]}
    assert [s["id"] for s in d["storeys"]] == ["L0", "L1"] and by["EG"]["id"] == "L0", d["storeys"]
    assert by["EG"]["height"] == 3.0 and by["EG"]["n_spaces"] == 2 and by["EG"]["net_floor_area"] == 31.2
    assert by["EG"]["counts"]["walls"] == 5 and by["EG"]["counts"]["doors"] == 2 and by["EG"]["counts"]["windows"] == 2
    assert abs(by["EG"]["walls"]["length"] - (2 * 8.73 + 2 * 4 + 4)) < 1e-6
    assert by["EG"]["slabs"]["volume"] == 9.5

    sp = {r[3]: dict(zip(d["spaces"]["columns"], r)) for r in d["spaces"]["rows"]}
    assert sp["Wohnen"]["net_area"] == 20.0 and sp["Wohnen"]["height"] == 2.6
    assert "W5" in sp["Wohnen"]["walls"] and "W5" in sp["Bad"]["walls"], sp
    assert sp["Bad"]["contents"] == {"sanitary:WC": 1, "sanitary:Waschtisch": 1}, sp["Bad"]

    rels = {(r["rel"], r["a"], r.get("b"), r.get("via")) for r in d["relations"]}
    S1, S2 = sp["Wohnen"]["id"], sp["Bad"]["id"]
    assert ("adjacent", S1, S2, "W5") in rels, rels
    assert ("connected_via", S1, S2, "D1") in rels or ("connected_via", S2, S1, "D1") in rels, rels
    assert any(r[0] == "connected_via" and "OUTSIDE" in (r[1], r[2]) and r[3] == "D2" for r in rels), rels
    assert ("faces_outside", S1, None, "WN1") in rels and ("faces_outside", S2, None, "WN2") in rels, rels
    assert ("connects_storeys", "L0", "L1", "ST1") in rels, rels

    t = {k: [dict(zip(v["columns"], row)) for row in v["rows"]] for k, v in d["elements"].items()}
    assert t["doors"][0]["host"] == "W5" and t["doors"][0]["width"] == 0.89 or t["doors"][0]["width"] == 0.88
    assert t["windows"][0]["area"] == 1.8 and t["windows"][0]["facade"] == "N", t["windows"][0]
    assert t["windows"][1]["facade"] == "E"
    assert t["stairs"][0]["risers"] == 17 and t["stairs"][0]["storeys"] == ["L0", "L1"]
    assert t["columns"][0]["section"] == "300x300 mm" and t["columns"][0]["length"] == 3.0
    assert t["walls"][0]["ext"] is True and t["walls"][4]["load_bearing"] is False

    types = {x["name"]: x for x in d["types"]}
    assert types["AW 36.5 Ziegel"]["count"] == 5 and types["AW 36.5 Ziegel"]["thickness"] == 0.365
    assert types["AW 36.5 Ziegel"]["props"]["FireRating"] == "REI90"
    assert types["Stuhl"]["count"] == 2, types.keys()

    s = d["summary"]
    assert s["counts"]["sanitary"] == 2 and s["counts"]["furniture"] == 2 and s["counts"]["storeys"] == 2
    assert s["walls"]["external"] == 5 and s["walls"]["internal"] == 1
    assert s["doors"]["width_max"] == 1.01 and s["doors"]["external"] == 1 and s["doors"]["fire_rated"] == 1
    assert s["windows"]["by_facade"] == {"N": 1, "E": 1}
    assert d["mep"]["systems"][0]["segment_length"] == 10.5 and d["mep"]["by_class"]["IfcPipeSegment"]["length"] == 10.5
    assert [(x["space"], x["category"]) for x in t["furniture"]] == [(S1, "seat")] * 2, t["furniture"]
    assert {x["category"] for x in t["sanitary"]} == {"toilet", "washbasin"} and all(x["space"] == S2 for x in t["sanitary"])
    assert sp["Bad"]["categories"] == ["bathroom"] and sp["Schlafen"]["categories"] == ["bedroom"] and sp["Bad"]["use"] == "restroom"
    assert "unit" not in d["spaces"]["columns"], d["spaces"]["columns"]          # all-null columns are dropped
    f = d["facts"]
    assert f["counts"]["toilets"] == 1 and f["counts"]["washbasins"] == 1 and f["counts"]["sinks_and_washbasins"] == 1
    assert f["counts"]["bathrooms"] == 1 and f["counts"]["bedrooms"] == 1 and f["counts"]["living_rooms"] == 1
    assert f["counts"]["doors"] == 2 and f["counts"]["doors_external"] == 1 and f["counts"]["windows"] == 2
    assert f["areas"]["net_floor_area"] == 63.2 and f["areas"]["by_name"]["Bad"]["net_area"] == 11.2, f["areas"]
    assert d["schema_version"] == "building-graph/1" and d["source"]["kind"] == "ifc"
    gfa = by["EG"]["gross_floor_area"]
    assert abs(gfa - 8.73 * 4.73) < 1.0, gfa

    # no geometry: still valid, counts and areas intact
    raw2 = model(); raw2["shapes"] = {}
    d2 = json.loads(build_graph(raw2).model_dump_json(exclude_none=True))
    assert d2["summary"]["counts"]["doors"] == 2 and not any(r["rel"] == "adjacent" for r in d2["relations"])
    # real exports: closed solids (floor AND ceiling triangles), georeferenced coordinates, areas only as
    # custom properties / quantities -> areas from the geometry must not shrink, stated areas win
    raw3 = model()
    for e in raw3["elements"]:
        if e["cls"] == "IfcSpace":
            e["psets"] = {}
    sp3 = {e["long_name"]: e for e in raw3["elements"] if e["cls"] == "IfcSpace"}
    sp3["Bad"]["psets"] = {"AR_Ruum": {"100_Nimi": "Bad", "120_Pindala": "11,2"}}           # ArchiCAD, text value
    sp3["Schlafen"]["psets"] = {"GSA Space Areas": {"GSA BIM Area": 31.5}}                  # Revit quantity set
    for sh in raw3["shapes"].values():
        t = [[[x + 538497.37, y + 6591600.81] for x, y in tr] for tr in sh["tris"]]
        sh["tris"] = t + [list(reversed(tr)) for tr in t]                                  # bottom + top face
    d3 = json.loads(build_graph(raw3).model_dump_json(exclude_none=True))
    sp = {r[3]: dict(zip(d3["spaces"]["columns"], r)) for r in d3["spaces"]["rows"]}
    assert abs(sp["Wohnen"]["net_area"] - 20.0) < 0.6 and sp["Wohnen"]["area_src"] == "geometry", sp["Wohnen"]
    assert sp["Bad"]["net_area"] == 11.2 and sp["Bad"]["area_src"] == "AR_Ruum.120_Pindala", sp["Bad"]
    assert sp["Schlafen"]["net_area"] == 31.5 and sp["Schlafen"]["area_src"] == "GSA Space Areas.GSA BIM Area"
    assert abs(sp["Wohnen"]["centroid_xy"][0] - (538497.37 + 2.5)) < 0.1, sp["Wohnen"]["centroid_xy"]   # world coords
    rels3 = {(r["rel"], r["a"], r.get("b"), r.get("via")) for r in d3["relations"]}
    assert ("adjacent", sp["Wohnen"]["id"], sp["Bad"]["id"], "W5") in rels3, rels3          # walls found again
    assert any("area_src" in n for n in d3["notes"])
    # overlapping gross-area space, IsExternal missing, beam over a room, NetFootprintArea, untyped names
    raw5 = model()
    G5 = 1
    for e in raw5["elements"]:
        if e["cls"] == "IfcWall":
            e["psets"] = {k: {kk: vv for kk, vv in v.items() if kk != "IsExternal"} for k, v in e["psets"].items()}
        if e["cls"] == "IfcDoor":
            e["psets"] = {}
        if e["cls"] == "IfcSpace" and e["long_name"] == "Bad":
            e["psets"] = {None: {"NetFootprintArea": 11.0}}
    brutto = el("IfcSpace", G5, "70.001", {}, long_name="Bruttoareal EG")
    raw5["elements"].append(brutto)
    raw5["shapes"][brutto["eid"]] = rect(-0.365, -0.365, 8.365, 4.365, 0, 3)
    beam = el("IfcBeam", G5, "HEB 200")
    raw5["elements"].append(beam)
    raw5["shapes"][beam["eid"]] = rect(2.0, -0.2, 2.2, 4.2, 2.6, 2.8)
    for i, n in enumerate(["T01-0", "T01-1", "T03"]):
        st5 = el("IfcStair", G5, n)
        raw5["elements"].append(st5)
    d5 = json.loads(build_graph(raw5).model_dump_json(exclude_none=True))
    sp5 = {r[3]: dict(zip(d5["spaces"]["columns"], r)) for r in d5["spaces"]["rows"]}
    assert abs(sp5["Bruttoareal EG"]["net_area"] - 8.73 * 4.73) < 1.0, sp5["Bruttoareal EG"]   # not carved by rooms
    assert {sp5["Wohnen"]["id"], sp5["Bad"]["id"]} <= set(sp5["Bruttoareal EG"]["contains"]), sp5["Bruttoareal EG"]
    assert not any(sp5["Bruttoareal EG"]["id"] in (r["a"], r.get("b")) for r in d5["relations"] if r["rel"] == "adjacent")
    assert abs(sp5["Wohnen"]["net_area"] - 20.0) < 0.6 or sp5["Wohnen"]["area_src"] != "geometry"
    assert sp5["Bad"]["net_area"] == 11.0 and sp5["Bad"]["area_src"].endswith("NetFootprintArea"), sp5["Bad"]
    t5 = {k: [dict(zip(v["columns"], row)) for row in v["rows"]] for k, v in d5["elements"].items()}
    w5 = {w["id"]: w for w in t5["walls"] if w["storey"] == "L0"}
    assert [w5[f"W{i}"]["ext"] for i in range(1, 6)] == [True, True, True, True, False], w5
    assert any("IsExternal missing" in n for n in d5["notes"])
    assert t5["doors"][1]["ext"] is True and t5["doors"][0]["ext"] is False, t5["doors"]   # via OUTSIDE
    assert sp5["Wohnen"]["id"] in t5["beams"][0]["spaces"] and sp5["Bad"]["id"] not in t5["beams"][0]["spaces"]
    tn = {t["name"] for t in d5["types"] if t["ifc_class"] == "IfcStair"}
    assert {"T01", "T03"} <= tn and "T" not in tn, tn
    assert "IfcSanitaryTerminal" in d5["mep"]["by_class"]
    assert d5["mep"]["by_domain"]["plumbing"]["by_class"] == {"IfcPipeSegment": 3, "IfcSanitaryTerminal": 2}
    share = t5["beams"][0]["spaces"][sp5["Wohnen"]["id"]]
    assert 0.8 <= share <= 1.0, t5["beams"][0]                                             # beam crosses Wohnen
    assert w5["W1"]["ext_src"].startswith("geometry") and "ext_src" not in d["elements"]["walls"]["columns"], w5["W1"]
    assert t5["doors"][1]["ext_src"] == "geometry: leads outside"
    st5 = {x["name"]: x for x in d5["storeys"]}
    assert st5["EG"]["gross_floor_area_src"].startswith("estimate"), st5["EG"]

    # old exports (hitos, 2006): property sets and types without a name must not break the conversion
    raw4 = model()
    for e in raw4["elements"]:
        e["psets"] = {None: {"IsExternal": True}, **e["psets"]}
        if e.get("type"):
            e["type"] = {**e["type"], "name": None}
    d4 = json.loads(build_graph(raw4).model_dump_json(exclude_none=True))
    assert d4["facts"]["counts"]["doors"] == 2 and d4["types"], "unnamed sets / types"
    print(f"ok  ({len(json.dumps(d)) / 1024:.1f} KB, {len(d['relations'])} relations)")


if __name__ == "__main__":
    main()
