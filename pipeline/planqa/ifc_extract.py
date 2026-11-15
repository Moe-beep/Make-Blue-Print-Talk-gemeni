"""Raw IFC dicts (planqa.ifc_api.read_ifc) -> condensed IfcGraph for an LLM agent.

Pure Python + numpy/OpenCV, no ifcopenshell, so it is testable with synthetic data.
Room relations (adjacent / door connects / window faces outside / wall bounds space)
come from IfcRelSpaceBoundary where present and from a 5 cm raster of each storey's
footprints otherwise; every relation says which source it came from.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import cv2
import numpy as np

from .building_schema import (OUTDOOR, compute_facts, furniture_category, make_table, sanitary_category,
                              space_categories, space_use)
from .ifc_schema import Building, IfcGraph, Relation, Source, Storey, Table, TypeInfo

# group -> (class prefixes, id prefix). Order matters (flights before stairs).
GROUPS = [
    ("walls", ("IfcWall",), "W"), ("curtain_walls", ("IfcCurtainWall",), "CW"), ("doors", ("IfcDoor",), "D"),
    ("windows", ("IfcWindow",), "WN"), ("slabs", ("IfcSlab",), "SL"), ("roofs", ("IfcRoof",), "R"),
    ("columns", ("IfcColumn",), "C"), ("beams", ("IfcBeam",), "B"), ("members", ("IfcMember",), "M"),
    ("stairs", ("IfcStair",), "ST"), ("ramps", ("IfcRamp",), "RA"), ("transport", ("IfcTransportElement",), "TR"),
    ("railings", ("IfcRailing",), "RL"), ("coverings", ("IfcCovering",), "CV"), ("footings", ("IfcFooting", "IfcPile"), "FT"),
]
PARTS = ("IfcStairFlight", "IfcRampFlight", "IfcBuildingElementPart")   # counted via their parent
LINEAR_MEP = ("IfcPipeSegment", "IfcDuctSegment", "IfcCableSegment", "IfcCableCarrierSegment")
COLUMNS = {
    "walls": ["id", "guid", "name", "type", "storey", "ext", "load_bearing", "length", "thickness", "height",
              "net_side_area", "gross_side_area", "volume", "spaces", "ext_src"],
    "curtain_walls": ["id", "guid", "name", "type", "storey", "ext", "length", "height", "area"],
    "doors": ["id", "guid", "name", "type", "storey", "width", "height", "fire_rating", "ext", "operation", "host", "connects",
              "ext_src"],
    "windows": ["id", "guid", "name", "type", "storey", "width", "height", "area", "ext", "host", "spaces", "facade"],
    "slabs": ["id", "guid", "name", "type", "storey", "kind", "area", "volume", "thickness"],
    "roofs": ["id", "guid", "name", "type", "storey", "kind", "area"],
    "columns": ["id", "guid", "name", "type", "storey", "load_bearing", "length", "volume", "section"],
    "beams": ["id", "guid", "name", "type", "storey", "load_bearing", "length", "volume", "section", "spaces"],
    "members": ["id", "guid", "name", "type", "storey", "kind", "length", "section"],
    "stairs": ["id", "guid", "name", "type", "storey", "flights", "risers", "riser_height", "tread_length", "storeys"],
    "ramps": ["id", "guid", "name", "type", "storey", "flights", "storeys"],
    "transport": ["id", "guid", "name", "type", "storey", "kind", "storeys"],
    "railings": ["id", "guid", "name", "type", "storey", "length", "height"],
    "coverings": ["id", "guid", "name", "type", "storey", "space", "kind", "area"],
    "footings": ["id", "guid", "name", "type", "storey", "kind", "volume"],
    "sanitary": ["id", "guid", "name", "type", "storey", "space", "category"],
    "furniture": ["id", "guid", "name", "type", "storey", "space", "category"],
}
EXTRA_PREFIX = {"sanitary": "SA", "furniture": "FU"}
SPACE_COLUMNS = ["id", "guid", "number", "name", "storey", "categories", "use", "unit", "outdoor", "object_type",
                 "net_area", "area_src", "gross_area", "height", "volume", "perimeter", "centroid_xy", "walls",
                 "contents", "contains"]
SC = {c: i for i, c in enumerate(SPACE_COLUMNS)}   # column index by name


def r(x, n=2):
    return None if x is None else round(float(x), n)


def q(e: dict, *names, sets=None):
    """First numeric quantity/property with one of `names` (Qto_* first, then any pset)."""
    psets = e.get("psets", {})
    order = sorted(psets, key=lambda s: (not (str(s).startswith("Qto_") or "Quantit" in str(s) or s == "BaseQuantities"), str(s)))
    for n in names:
        for s in order:
            if sets and not any(k in s for k in sets):
                continue
            v = psets[s].get(n)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


# Area properties outside the standard quantities, e.g. Revit 'GSA BIM Area', ArchiCAD 'Pindala' (et) / 'Fläche' as text
AREA_NAME = re.compile(r"(?i)(\barea\b|area$|fl(ä|ae)che|pindala|\bareal|pinta.?ala|superficie|surface|oppervlakte|"
                       r"powierzchnia|plocha)")
AREA_SKIP = re.compile(r"(?i)(wall|window|door|opening|perimeter|volume|height|ceiling|gross|brutto|outer|external|"
                       r"side|finish|cover|boundary|roof)")


def _number(v) -> Optional[float]:
    """5.1 / '5,1' / '5.10 m²' -> 5.1; anything else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*(-?\d+(?:[.,]\d+)?)\s*(m2|m²|sqm)?\s*", str(v), re.I) if isinstance(v, str) else None
    return float(m[1].replace(",", ".")) if m else None


def space_area(e: dict) -> tuple[Optional[float], Optional[str]]:
    """Net area of a space as stated in the model, with its source: standard quantities first (NetFloorArea,
    NetArea), then any property / quantity named like an area (any language, text values too), then
    GrossFloorArea. None -> the caller falls back to the geometry."""
    psets = e.get("psets", {})
    order = sorted(psets, key=lambda s: (not (str(s).startswith("Qto_") or "Quantit" in str(s) or s == "BaseQuantities"), str(s)))
    for names in (("NetFloorArea", "NetArea", "NetFootprintArea"), None, ("GrossFloorArea", "GrossFootprintArea")):
        for s in order:
            props = psets[s]
            keys = [n for n in names if n in props] if names else \
                sorted((k for k in props if isinstance(k, str) and AREA_NAME.search(k) and not AREA_SKIP.search(k)),
                       key=lambda k: "net" not in k.lower())
            for k in keys:
                v = _number(props[k])
                if v and v > 0:
                    return v, f"{s}.{k}"
    return None, None


MEP_DOMAINS = [   # (domain, pattern on IFC class / type class / predefined type / name)
    ("plumbing", r"Sanitary|WasteTerminal|Pipe|Valve|Pump|Tank|Interceptor|StackTerminal|Toilet|Basin|Sink|Shower|Bath"),
    ("ventilation", r"AirTerminal|Duct|Fan|Damper|AirToAir|Filter|Ventil"),
    ("heating", r"SpaceHeater|Boiler|Radiator|Convector|HeatExchanger|HeatPump|Burner|Heating"),
    ("cooling", r"Chiller|CooledBeam|CoolingTower|Condenser|Evaporator|Cooling"),
    ("lighting", r"Light|Lamp|Luminaire"),
    ("electrical", r"Electric|Outlet|Switch|Cable|Transformer|Protective|Distribution(Board|Panel)|Junction|Motor|Generator"),
    ("fire_protection", r"FireSuppression|Sprinkler|Alarm|Detector|Hydrant"),
    ("communication", r"Communication|Audio|Network|Antenna"),
]


def mep_domain(cls: str, type_cls: Optional[str], predefined: Optional[str], name: Optional[str]) -> str:
    """MEP domain from class, type class, predefined type, name (in this order)."""
    for text in (type_cls, cls, predefined, name):
        if not text:
            continue
        for d, rx in MEP_DOMAINS:
            if re.search(rx, str(text), re.I):
                return d
    return "other"


def common(e: dict, prop: str):
    """Value of a Pset_*Common property (IsExternal, LoadBearing, FireRating, ...)."""
    for s, props in e.get("psets", {}).items():
        if str(s).startswith("Pset_") and str(s).endswith("Common") and props.get(prop) not in (None, ""):
            return props[prop]
    return None


def group_of(e: dict) -> Optional[str]:
    c = e["cls"]
    if c == "IfcSpace" or c.startswith(PARTS):
        return None
    if c == "IfcSanitaryTerminal" or (e.get("type") or {}).get("cls") == "IfcSanitaryTerminalType":
        return "sanitary"
    for g, prefixes, _ in GROUPS:
        if c.startswith(prefixes):
            return g
    if c.startswith(("IfcFurnishingElement", "IfcFurniture", "IfcSystemFurnitureElement")):
        return "furniture"
    if e.get("is_mep"):
        return "mep"
    return "other"


def base_name(e: dict) -> str:
    t = e.get("type") or {}
    n = t.get("name") or e.get("object_type") or e.get("name") or e["cls"]
    if t.get("name"):
        return n
    n = re.sub(r"\s*[:_-]\s*\d+$", "", n) or n     # instance number: Revit "Family:Type:123456", "T01-0" -> "T01"
    parts = list(dict.fromkeys(p.strip() for p in n.split(":") if p.strip()))
    return ":".join(parts) or n


# ======================================================================== main

def build_graph(raw: dict) -> IfcGraph:
    notes: list[str] = []
    els = raw["elements"]
    by_eid = {e["eid"]: e for e in els}
    shapes = raw.get("shapes") or {}

    # ---- storeys
    st_raw = sorted(raw["storeys"], key=lambda s: (s["elevation"] is None, s["elevation"] or 0))
    sid = {s["eid"]: f"L{i}" for i, s in enumerate(st_raw)}
    storeys = []
    for i, s in enumerate(st_raw):
        nxt = st_raw[i + 1]["elevation"] if i + 1 < len(st_raw) else None
        h = (nxt - s["elevation"]) if (nxt is not None and s["elevation"] is not None) else q(s, "Height", "GrossHeight")
        storeys.append(Storey(id=sid[s["eid"]], name=s["name"], elevation=r(s["elevation"], 3), height=r(h, 3),
                              gross_floor_area=r(q(s, "GrossFloorArea", "GrossArea"))))
    st_by_id = {s.id: s for s in storeys}
    elev = {s.id: s.elevation for s in storeys}

    # ---- spaces
    spaces = [e for e in els if e["cls"] == "IfcSpace"]
    spid = {e["eid"]: f"S{i}" for i, e in enumerate(spaces, 1)}

    # ---- groups, ids
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in els:
        g = group_of(e)
        if g:
            grouped[g].append(e)
    prefix = {**{g: p for g, _, p in GROUPS}, **EXTRA_PREFIX}
    eid2id: dict[int, str] = dict(spid)
    for g, items in grouped.items():
        if g in prefix:
            for i, e in enumerate(items, 1):
                eid2id[e["eid"]] = f"{prefix[g]}{i}"

    def storey_of(e):
        return sid.get(e.get("storey_eid"))

    # ---- types (one entry per type object, or per class + base name if untyped)
    types: dict[Any, TypeInfo] = {}
    type_of: dict[int, str] = {}
    members: dict[Any, list[dict]] = defaultdict(list)
    for g, items in grouped.items():
        for e in items:
            t = e.get("type")
            key = ("T", t["eid"]) if t else ("N", g, base_name(e))
            members[key].append(e)
    for i, (key, items) in enumerate(members.items(), 1):
        e0, t = items[0], items[0].get("type") or {}
        mats = list(dict.fromkeys(m for e in items for m in (e.get("material") or {}).get("materials", []) if m))
        layers = next(((e.get("material") or {}).get("layers") for e in items if (e.get("material") or {}).get("layers")), [])
        thk = sum(l[1] or 0 for l in layers) if layers else None
        if thk is None and e0["cls"].startswith(("IfcWall", "IfcSlab", "IfcCovering")):
            vals = [q(e, "Width", "Thickness", "Depth") for e in items]
            vals = [v for v in vals if v]
            thk = Counter(round(v, 3) for v in vals).most_common(1)[0][0] if vals else None
        props: dict[str, Any] = {}
        all_props = defaultdict(list)
        for e in items:
            for s, ps in e.get("psets", {}).items():
                if str(s).startswith("Pset_") and str(s).endswith("Common"):
                    for k, v in ps.items():
                        if v not in (None, "") and not (isinstance(v, str) and len(v) > 80):
                            all_props[k].append(v)
        for k, vs in all_props.items():
            c = Counter(map(str, vs))
            props[k] = vs[0] if len(c) == 1 else "varies: " + ", ".join(f"{v} ({n})" for v, n in c.most_common(4))
        ti = TypeInfo(id=f"T{i}", ifc_class=e0["cls"], name=base_name(e0), count=len(items),
                      predefined=t.get("predefined") or e0.get("predefined"), materials=mats, layers=layers,
                      thickness=r(thk, 3), profile=next((e["profile"] for e in items if e.get("profile")), None),
                      props=props)
        types[key] = ti
        for e in items:
            type_of[e["eid"]] = ti.id

    # ---- geometry: footprints per storey -> relations
    rel = _relations(spaces, grouped, shapes, raw.get("space_boundaries") or [], eid2id, sid, elev, notes)
    space_walls, door_conn, window_spaces, item_space, stair_storeys, sp_geo, st_gross = rel["parts"]
    relations = rel["relations"]

    # ---- inventory (MEP / other aggregated; sanitary and furniture become element tables)
    inventory: dict[str, Any] = {}
    contents: dict[str, Counter] = defaultdict(Counter)
    for g in ("furniture", "sanitary", "mep", "other"):
        agg: dict[str, dict] = {}
        for e in grouped.get(g, []):
            tid = type_of[e["eid"]]
            a = agg.setdefault(tid, {"name": next(t.name for t in types.values() if t.id == tid), "count": 0,
                                     "by_storey": Counter(), "by_space": Counter()})
            a["count"] += 1
            st = storey_of(e)
            if st:
                a["by_storey"][st] += 1
            sp = spid.get(e.get("space_eid")) or item_space.get(e["eid"])
            if sp:
                a["by_space"][sp] += 1
                if g in ("furniture", "sanitary"):
                    contents[sp][f"{g}:{a['name']}"] += 1
        if agg and g in ("mep", "other"):
            inventory[g] = {k: {**v, "by_storey": dict(v["by_storey"]), "by_space": dict(v["by_space"])} for k, v in agg.items()}

    # ---- space table
    sp_rows, area_mismatch = [], []
    for e in spaces:
        s = spid[e["eid"]]
        g = sp_geo.get(s, {})
        net, area_src = space_area(e)
        if net is None and g.get("area"):
            net, area_src = g["area"], "geometry"
        elif net is not None and g.get("area") and abs(g["area"] - net) > 0.15 * net + 0.5:
            area_mismatch.append(s)
        otype = e.get("object_type") or common(e, "Category")
        cats = space_categories(e.get("long_name"), e.get("name"), otype)
        outdoor = bool(set(cats) & OUTDOOR) or common(e, "IsExternal") is True
        sp_rows.append([s, e["guid"], e.get("name"), e.get("long_name"), storey_of(e), cats or None, space_use(cats),
                        e.get("unit"), outdoor or None, otype, r(net), area_src, r(q(e, "GrossFloorArea")),
                        r(q(e, "Height", "FinishCeilingHeight", "NetHeight") or g.get("height")),
                        r(q(e, "NetVolume", "GrossVolume")), r(q(e, "NetPerimeter", "GrossPerimeter") or g.get("perimeter")),
                        g.get("centroid"), sorted(space_walls.get(s, []), key=_idkey) or None,
                        dict(contents[s]) or None, g.get("contains")])
    space_table = Table(columns=SPACE_COLUMNS, rows=sp_rows)
    srcs = Counter((row[SC["area_src"]] or "none").split(".")[-1] if row[SC["area_src"]] != "geometry" else "geometry"
                   for row in sp_rows)
    if sp_rows:
        notes.append("Space net areas by source (area_src): " + ", ".join(f"{k} {v}" for k, v in srcs.most_common())
                     + ". Stated values (quantities / properties) win over the geometry.")
    if area_mismatch:
        notes.append(f"{len(area_mismatch)} space(s) whose stated area differs > 15 % from the geometry "
                     f"(e.g. {', '.join(area_mismatch[:5])}): the stated value is used.")
    area_by_space = {row[0]: row[SC["net_area"]] for row in sp_rows}

    # ---- element tables
    inferred, length_src = Counter(), Counter()
    facade_of = rel["facade"]
    tables = {}
    for g, items in grouped.items():
        if g not in COLUMNS:
            continue
        rows = []
        for e in items:
            i = eid2id[e["eid"]]
            base = [i, e["guid"], e.get("name"), type_of[e["eid"]], storey_of(e)]
            ext, lb = common(e, "IsExternal"), common(e, "LoadBearing")
            ext_src = None   # None = IsExternal of the model; otherwise how it was derived
            if ext is None and g == "walls" and e["eid"] in rel["ext_geo"]:   # IsExternal missing: from geometry
                ext, ext_src = rel["ext_geo"][e["eid"]], "geometry: touches the outside of the storey outline"
                inferred["walls"] += 1
            elif ext is None and g == "doors" and door_conn.get(i):
                ext, ext_src = "OUTSIDE" in door_conn[i], "geometry: leads outside"
                inferred["doors"] += 1
            geo = rel["dims"].get(e["eid"], {})
            if g == "walls":
                length_src[("model" if q(e, "Length") else "geometry" if geo.get("length") else "none")] += 1
                row = base + [ext, lb, r(q(e, "Length") or geo.get("length")),
                              r(q(e, "Width", "Thickness") or types_by_id(types)[type_of[e["eid"]]].thickness or geo.get("width"), 3),
                              r(q(e, "Height") or geo.get("height")), r(q(e, "NetSideArea")), r(q(e, "GrossSideArea")),
                              r(q(e, "NetVolume", "GrossVolume"), 3), sorted(set(rel["wall_spaces"].get(i, [])), key=_idkey) or None,
                              ext_src]
            elif g == "curtain_walls":
                row = base + [ext, r(q(e, "Length") or geo.get("length")), r(q(e, "Height") or geo.get("height")),
                              r(q(e, "GrossSideArea", "NetSideArea", "Area"))]
            elif g == "doors":
                row = base + [r(e["attrs"].get("OverallWidth") or q(e, "Width")), r(e["attrs"].get("OverallHeight") or q(e, "Height")),
                              common(e, "FireRating"), ext, e["attrs"].get("OperationType") or common(e, "OperationType"),
                              eid2id.get(e.get("host_eid")), door_conn.get(i), ext_src]
            elif g == "windows":
                w, h = e["attrs"].get("OverallWidth") or q(e, "Width"), e["attrs"].get("OverallHeight") or q(e, "Height")
                row = base + [r(w), r(h), r(q(e, "Area") or (w * h if w and h else None)), ext,
                              eid2id.get(e.get("host_eid")), window_spaces.get(i), facade_of.get(i)]
            elif g == "slabs":
                row = base + [e.get("predefined") or (e.get("type") or {}).get("predefined"),
                              r(q(e, "NetArea", "GrossArea", "Area")), r(q(e, "NetVolume", "GrossVolume"), 3),
                              r(q(e, "Depth", "Width", "Thickness") or types_by_id(types)[type_of[e["eid"]]].thickness, 3)]
            elif g == "roofs":
                row = base + [e.get("predefined"), r(q(e, "NetArea", "GrossArea", "Area"))]
            elif g in ("columns", "beams"):
                p = e.get("profile") or types_by_id(types)[type_of[e["eid"]]].profile
                row = base + [lb, r(q(e, "Length") or (p or {}).get("extrusion")), r(q(e, "NetVolume", "GrossVolume"), 3), _section(p)]
                if g == "beams":   # spaces the beam passes over (geometry): {space: share of the beam above it}
                    bs = rel["beam_spaces"].get(e["eid"]) or {}
                    row.append(dict(sorted(bs.items(), key=lambda x: -x[1])) or None)
            elif g == "members":
                p = e.get("profile")
                row = base + [e.get("predefined"), r(q(e, "Length") or (p or {}).get("extrusion")), _section(p)]
            elif g in ("stairs", "ramps"):
                fl = e.get("flights") or []
                risers = sum(f["risers"] or 0 for f in fl) or e["attrs"].get("NumberOfRisers") or e["attrs"].get("NumberOfRiser") \
                    or common(e, "NumberOfRiser")
                extra = [len(fl) or None, risers or None,
                         r(next((f["riser_height"] for f in fl if f.get("riser_height")), None) or common(e, "RiserHeight"), 3),
                         r(next((f["tread_length"] for f in fl if f.get("tread_length")), None) or common(e, "TreadLength"), 3)]
                row = base + (extra if g == "stairs" else extra[:1]) + [stair_storeys.get(i)]
            elif g == "transport":
                row = base + [e.get("predefined"), stair_storeys.get(i)]
            elif g == "railings":
                row = base + [r(q(e, "Length")), r(q(e, "Height") or common(e, "Height"))]
            elif g in ("sanitary", "furniture"):
                tname = types_by_id(types)[type_of[e["eid"]]].name
                t = e.get("type") or {}
                cat = sanitary_category(e.get("predefined"), t.get("predefined"), tname, e.get("name")) if g == "sanitary" \
                    else furniture_category(tname, e.get("name"), e.get("object_type"))
                row = base + [spid.get(e.get("space_eid")) or item_space.get(e["eid"]), cat]
            elif g == "coverings":
                row = base + [spid.get(e.get("space_eid")) or item_space.get(e["eid"]),
                              e.get("predefined") or (e.get("type") or {}).get("predefined"), r(q(e, "NetArea", "GrossArea", "Area"))]
            else:  # footings
                row = base + [e.get("predefined"), r(q(e, "NetVolume", "GrossVolume"), 3)]
            rows.append(row)
        tables[g] = Table(columns=COLUMNS[g], rows=rows)

    if length_src.get("geometry"):
        notes.append(f"Wall lengths: {length_src['model']} from model quantities, {length_src['geometry']} DERIVED "
                     "from the geometry (no length quantity in the model): say so when you use them.")
    if inferred:
        notes.append("IsExternal missing in the model, inferred from the geometry for "
                     + ", ".join(f"{n} {g}" for g, n in inferred.items())
                     + " (wall touching the outside of the storey outline / door leading outside).")

    # ---- MEP
    mep: dict[str, Any] = {}
    mep_items = grouped.get("mep", []) + grouped.get("sanitary", [])   # sanitary terminals are plumbing flow terminals
    if grouped.get("sanitary"):
        notes.append("mep.by_class also counts the sanitary terminals (plumbing flow terminals); "
                     "they are listed in elements.sanitary.")
    if mep_items:
        by_cls: dict[str, dict] = {}
        for e in mep_items:
            c = by_cls.setdefault(e["cls"], {"count": 0, "by_storey": Counter()})
            c["count"] += 1
            if storey_of(e):
                c["by_storey"][storey_of(e)] += 1
            if e["cls"].startswith(LINEAR_MEP):
                ln = q(e, "Length") or (e.get("profile") or {}).get("extrusion")
                if ln:
                    c["length"] = c.get("length", 0) + ln
        mep["by_class"] = {k: {**v, "by_storey": dict(v["by_storey"]), **({"length": r(v["length"])} if "length" in v else {})}
                           for k, v in sorted(by_cls.items(), key=lambda x: -x[1]["count"])}
        sysl = []
        for s in raw.get("systems") or []:
            comp = Counter(by_eid[m]["cls"] for m in s["members"] if m in by_eid)
            if comp:
                ln = sum((q(by_eid[m], "Length") or 0) for m in s["members"] if m in by_eid and by_eid[m]["cls"].startswith(LINEAR_MEP))
                sysl.append({"name": s["name"], "kind": s["kind"], "components": sum(comp.values()),
                             "by_class": dict(comp), **({"segment_length": r(ln)} if ln else {})})
        mep["systems"] = sysl
        # domain of every component from its IFC class / type (works without IfcSystem groups)
        dom: dict[str, Counter] = defaultdict(Counter)
        for e in mep_items:
            t = e.get("type") or {}
            dom[mep_domain(e["cls"], t.get("cls"), e.get("predefined") or t.get("predefined"), base_name(e))][
                (t.get("cls") or e["cls"]).replace("Type", "")] += 1
        mep["by_domain"] = {d: {"count": sum(c.values()), "by_class": dict(c)} for d, c in
                            sorted(dom.items(), key=lambda x: -sum(x[1].values()))}
        if not sysl:
            notes.append("No IfcSystem groups in the model: mep.by_domain is DERIVED from the IFC class / type of "
                         "each component (e.g. sanitary terminal -> plumbing); say so when you use it.")

    # ---- storey aggregates
    for row in sp_rows:
        st = st_by_id.get(row[4])
        if st:
            st.n_spaces += 1
            if not row[SC["outdoor"]]:
                st.net_floor_area = r(st.net_floor_area + (row[SC["net_area"]] or 0))
    for g, t in tables.items():
        ci = t.columns.index("storey")
        for row in t.rows:
            if row[ci] in st_by_id:
                st_by_id[row[ci]].counts[g] = st_by_id[row[ci]].counts.get(g, 0) + 1
    for g in ("mep", "other"):
        for a in inventory.get(g, {}).values():
            for st, n in a["by_storey"].items():
                st_by_id[st].counts[g] = st_by_id[st].counts.get(g, 0) + n
    if "walls" in tables:
        t = tables["walls"]
        for row in t.rows:
            st = st_by_id.get(row[4])
            if st:
                for k, ci in (("length", 7), ("net_side_area", 10), ("gross_side_area", 11), ("volume", 12)):
                    if row[ci] is not None:
                        st.walls[k] = r(st.walls.get(k, 0) + row[ci], 3)
    if "slabs" in tables:
        for row in tables["slabs"].rows:
            st = st_by_id.get(row[4])
            if st:
                for k, ci in (("area", 6), ("volume", 7)):
                    if row[ci] is not None:
                        st.slabs[k] = r(st.slabs.get(k, 0) + row[ci], 3)
    for st in storeys:
        if st.gross_floor_area is None and st_gross.get(st.id):
            st.gross_floor_area = st_gross[st.id]
            st.gross_floor_area_src = "estimate: outline of spaces and walls (not in the model)"
            notes.append(f"Gross floor area of {st.id} is an ESTIMATE from the outline of its spaces and walls "
                         "(the model states none): call it an estimate.")
        elif st.gross_floor_area is not None and st.gross_floor_area_src is None:
            st.gross_floor_area_src = "model"

    summary = _summary(storeys, space_table, tables, types, inventory, area_by_space, raw["building"])
    b = raw["building"]
    building = Building(project=b.get("project"), project_long_name=b.get("project_long_name"), site=b.get("site"), name=b.get("name"), description=b.get("description"),
                        type=b.get("object_type") or b.get("occupancy"), address=b.get("address"),
                        latitude=b.get("lat"), longitude=b.get("lon"), georeferenced=bool(b.get("georef") or b.get("lat")),
                        true_north_deg=b.get("true_north_deg"))
    if not building.georeferenced:
        notes.append("Model is not georeferenced: compass directions (facade) are relative to project north.")
    h = raw["header"]
    keep = ("id", "storey")
    graph = IfcGraph(source=Source(file=h["file"], kind="ifc", ifc_schema=h["schema"], application=h.get("application"),
                                   view_definition=h.get("view_definition"), exported_at=h.get("exported_at"),
                                   extraction={"method": "ifc"}),
                     building=building, storeys=storeys, spaces=make_table(space_table.columns, space_table.rows, keep),
                     types=list(types.values()),
                     elements={g: make_table(t.columns, t.rows, keep) for g, t in tables.items()},
                     inventory=inventory, mep=mep, relations=relations, summary=summary)
    graph.facts = compute_facts(graph)
    if sp_rows and not any(row[SC["categories"]] for row in sp_rows):
        notes.append("No space name matched a room category: room-type counts in facts are 0, check spaces.name.")
    graph.notes = list(dict.fromkeys(notes))
    return graph


def types_by_id(types: dict) -> dict:
    return {t.id: t for t in types.values()}


def _idkey(s: str):
    m = re.match(r"([A-Z]+)(\d+)", s or "")
    return (m.group(1), int(m.group(2))) if m else (s or "", 0)


def _section(p: Optional[dict]) -> Optional[str]:
    if not p:
        return None
    if p.get("shape") == "rectangle":
        return f"{p['x'] * 1000:.0f}x{p['y'] * 1000:.0f} mm"
    if p.get("shape") == "circle":
        return f"Ø{p['d'] * 1000:.0f} mm"
    if p.get("shape") == "I":
        return f"I {p['width'] * 1000:.0f}x{p['depth'] * 1000:.0f} mm"
    return p.get("shape")


# ======================================================================== relations

def _relations(spaces, grouped, shapes, boundaries, eid2id, sid, storey_elev, notes) -> dict:
    """Space relations from IfcRelSpaceBoundary and/or a raster of the storey footprints."""
    space_walls, wall_spaces = defaultdict(set), defaultdict(set)
    door_conn, window_spaces, item_space, stair_storeys, sp_geo, st_gross, dims, facade = {}, {}, {}, {}, {}, {}, {}, {}
    ext_geo, beam_spaces = {}, defaultdict(dict)
    relations: list[Relation] = []
    seen = set()

    def add(relname, a, b=None, via=None, src="geometry"):
        key = (relname, *sorted([a, b or ""]), via)
        if key not in seen:
            seen.add(key)
            relations.append(Relation(rel=relname, a=a, b=b, via=via, src=src))

    walls = grouped.get("walls", []) + grouped.get("curtain_walls", [])
    doors, windows = grouped.get("doors", []), grouped.get("windows", [])
    beams = grouped.get("beams", [])
    ext = {e["eid"]: common(e, "IsExternal") for e in walls + doors + windows}

    # 1) IFC space boundaries (explicit, but adjacency across a long wall can be ambiguous)
    if boundaries:
        by_el = defaultdict(set)
        for sp, el, io in boundaries:
            s, x = eid2id.get(sp), eid2id.get(el)
            if not (s and x):
                continue
            by_el[x].add(s)
            if x.startswith(("W", "CW")):
                space_walls[s].add(x); wall_spaces[x].add(s)
        for x, ss in by_el.items():
            if x.startswith("D"):
                door_conn[x] = sorted(ss, key=_idkey)
            elif x.startswith("WN"):
                window_spaces[x] = sorted(ss, key=_idkey)
        notes.append("Space-wall/door/window links taken from IfcRelSpaceBoundary.")

    # 2) geometry raster per storey
    # local origin first: georeferenced models have coordinates like (538497, 6591600) m, where float32 only
    # resolves 0.5 m steps; geometry is shifted to the origin before it becomes float32
    first = next((s["tris"] for s in shapes.values() if len(s["tris"])), None)
    origin = np.asarray(first, np.float64).reshape(-1, 2)[0] if first is not None else np.zeros(2)
    tri = {eid: (np.asarray(s["tris"], np.float64) - origin).astype(np.float32)
           for eid, s in shapes.items() if len(s["tris"])}
    zr = {eid: (s["zmin"], s["zmax"]) for eid, s in shapes.items()}
    for e in walls:
        if e["eid"] in tri:
            pts = tri[e["eid"]].reshape(-1, 2)
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            z0, z1 = zr[e["eid"]]
            dims[e["eid"]] = {"length": max(w, h), "width": min(w, h), "height": z1 - z0}

    sp_items = [e for e in spaces if e["eid"] in tri]
    if not shapes:
        notes.append("No geometry: relations only from IfcRelSpaceBoundary (if any); no adjacency/facade analysis.")
    elif not sp_items:
        notes.append("Spaces have no geometry: no adjacency/facade analysis.")
    else:
        all_pts = np.concatenate([tri[e["eid"]].reshape(-1, 2) for e in sp_items])
        centre = all_pts.mean(0)
        storey_ids = sorted({e.get("storey_eid") for e in sp_items})
        for st_eid in storey_ids:
            st = sid.get(st_eid)
            S = [e for e in sp_items if e.get("storey_eid") == st_eid]
            Wl = [e for e in walls if e.get("storey_eid") == st_eid and e["eid"] in tri]
            Dr = [e for e in doors if e.get("storey_eid") == st_eid and e["eid"] in tri]
            Wn = [e for e in windows if e.get("storey_eid") == st_eid and e["eid"] in tri]
            pts = np.concatenate([tri[e["eid"]].reshape(-1, 2) for e in S + Wl + Dr + Wn])
            x0, y0 = pts.min(0) - 1.0
            x1, y1 = pts.max(0) + 1.0
            res = max(0.05, max(x1 - x0, y1 - y0) / 4000)
            Wpx, Hpx = int(math.ceil((x1 - x0) / res)) + 1, int(math.ceil((y1 - y0) / res)) + 1

            def px(t):
                return np.round((t - [x0, y0]) / res).astype(np.int32)

            def paint(target, items, start=1):
                for k, e in enumerate(items, start):
                    _fill(target, px(tri[e["eid"]]), k)

            S.sort(key=lambda e: -_tri_area(tri[e["eid"]]))
            sids = [eid2id[e["eid"]] for e in S]
            smap = np.zeros((Hpx, Wpx), np.int32)
            paint(smap, S)                              # larger first: smaller spaces end up on top
            # each space on its own (cropped to its box): true area even where other spaces overlap it
            own = {}
            for k, e in enumerate(S, 1):
                t = px(tri[e["eid"]])
                bx0, by0 = np.maximum(t.reshape(-1, 2).min(0), 0)
                bx1, by1 = np.minimum(t.reshape(-1, 2).max(0) + 1, [Wpx, Hpx])
                if bx1 <= bx0 or by1 <= by0:
                    continue
                m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                _fill(m, t - [bx0, by0], 1)
                if m.any():
                    own[k] = (bx0, by0, m)
            # a space that mostly lies under other spaces and holds >= 2 of them is an area / zone
            # (gross-area spaces like 'Bruttoareal', apartments drawn as spaces): no room for adjacency / doors
            containers = {}
            for k, (bx0, by0, m) in own.items():
                under = smap[by0:by0 + m.shape[0], bx0:bx0 + m.shape[1]][m > 0]
                others = Counter(under[(under != k) & (under > 0)].tolist())
                held = [j for j, n in others.items() if j in own and n >= 0.8 * own[j][2].sum()]
                if np.count_nonzero(under == k) < 0.5 * m.sum() and len(held) >= 2:
                    containers[k] = sorted((sids[j - 1] for j in held), key=_idkey)
            if containers:
                notes.append(f"{st}: {len(containers)} space(s) contain other spaces (gross-area / zone spaces, column "
                             "`contains`): they keep their area but are no rooms for adjacency and doors.")
                smap = np.zeros((Hpx, Wpx), np.int32)
                for k, e in enumerate(S, 1):
                    if k not in containers:
                        _fill(smap, px(tri[e["eid"]]), k)
            wmap = np.zeros((Hpx, Wpx), np.int32)
            paint(wmap, Wl)
            wids = [eid2id[e["eid"]] for e in Wl]
            wthk = [dims.get(e["eid"], {}).get("width", 0.2) for e in Wl]
            k1 = np.ones((3, 3), np.uint8)

            masks = {}
            for k, e in enumerate(S, 1):
                m = (smap == k).astype(np.uint8)
                if m.any():
                    masks[k] = m      # visible part: for adjacency
                if k not in own:
                    continue
                bx0, by0, m = own[k]
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                mo = cv2.moments(m, binaryImage=True)
                z0, z1 = zr[e["eid"]]
                sp_geo[sids[k - 1]] = {"area": float(m.sum()) * res * res, "perimeter": cv2.arcLength(c, True) * res,
                                       "centroid": [r(origin[0] + x0 + (bx0 + mo["m10"] / mo["m00"]) * res),
                                                    r(origin[1] + y0 + (by0 + mo["m01"] / mo["m00"]) * res)],
                                       "height": z1 - z0, **({"contains": containers[k]} if k in containers else {})}
            # wall bounds space; spaces adjacent across a wall or open
            near = int(math.ceil(0.15 / res))
            grown = {k: cv2.dilate(m, k1, iterations=near) for k, m in masks.items()}
            for k, g in grown.items():
                for w in np.unique(wmap[(g > 0) & (wmap > 0)]):
                    if np.count_nonzero((g > 0) & (wmap == w)) >= 3:
                        space_walls[sids[k - 1]].add(wids[w - 1]); wall_spaces[wids[w - 1]].add(sids[k - 1])
            ks = sorted(masks)
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    A, B = sids[a - 1], sids[b - 1]
                    shared = space_walls[A] & space_walls[B]
                    if np.count_nonzero(grown[a] & masks[b]) >= 0.5 / res:
                        add("adjacent", A, B, None)                         # touch directly (open plan / virtual boundary)
                        continue
                    best, best_n = None, 0
                    for w in shared:                                   # wall whose thickness separates a and b
                        if w not in wids:
                            continue
                        wm = (wmap == wids.index(w) + 1).astype(np.uint8)
                        it = int(math.ceil((wthk[wids.index(w)] + 0.2) / res))
                        through = cv2.dilate(grown[a] & wm, k1, iterations=it) & wm & grown[b]
                        n = int(np.count_nonzero(through))
                        if n > best_n:
                            best, best_n = w, n
                    if best and best_n * res * res >= 0.1:              # >= ~0.5 m of common wall
                        add("adjacent", A, B, best)
            # doors / windows
            far = int(math.ceil(0.35 / res))
            for e in Dr:
                d = eid2id[e["eid"]]
                m = np.zeros((Hpx, Wpx), np.uint8)
                _fill(m, px(tri[e["eid"]]), 1)
                m = cv2.dilate(m, k1, iterations=far)
                hits = Counter(smap[(m > 0) & (smap > 0)].tolist())
                conn = [sids[k - 1] for k, n in hits.most_common(2) if n >= 4]
                if len(conn) == 1:
                    conn.append("OUTSIDE")
                if conn:
                    door_conn.setdefault(d, conn)
            far = int(math.ceil(0.5 / res))
            for e in Wn:
                wi = eid2id[e["eid"]]
                m = np.zeros((Hpx, Wpx), np.uint8)
                _fill(m, px(tri[e["eid"]]), 1)
                m = cv2.dilate(m, k1, iterations=far)
                hits = Counter(smap[(m > 0) & (smap > 0)].tolist())
                ss = [sids[k - 1] for k, n in hits.most_common(2) if n >= 4]
                if ss:
                    window_spaces.setdefault(wi, ss)
                wp = tri[e["eid"]].reshape(-1, 2)
                c = wp.mean(0) - centre
                (_, _), (rw, rh), ang = cv2.minAreaRect(wp)
                a = math.radians(ang + (90 if rw > rh else 0))       # short axis = wall normal
                n = np.array([math.cos(a), math.sin(a)])
                if np.dot(n, c) < 0:
                    n = -n
                facade[wi] = _cardinal(math.degrees(math.atan2(n[0], n[1])))
            # gross footprint: spaces + walls, small gaps closed, holes filled
            fp = ((smap > 0) | (wmap > 0)).astype(np.uint8)
            fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, k1, iterations=int(0.3 / res))
            cnts, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            st_gross[st] = r(sum(cv2.contourArea(c) for c in cnts) * res * res)
            # walls without IsExternal: external if the wall touches the outside of the storey outline
            filled = np.zeros_like(fp)
            cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
            outside = filled == 0
            reach = max(2, int(math.ceil(0.1 / res)))
            for k, e in enumerate(Wl, 1):
                if ext.get(e["eid"]) is None:
                    wm = cv2.dilate((wmap == k).astype(np.uint8), k1, iterations=reach)
                    if wm.any():
                        ext_geo[e["eid"]] = bool(np.count_nonzero((wm > 0) & outside) >= 3)
            # beams over spaces: beams at the height of this storey's spaces (up to the ceiling + 0.6 m)
            zs = [zr[e["eid"]] for e in S]
            zlo, zhi = min(z[0] for z in zs), max(z[1] for z in zs) + 0.6
            min_px = max(4, int(0.25 / (res * res)))   # at least 0.25 m² of the beam over the space
            for e in beams:
                if e["eid"] in tri and zr[e["eid"]][1] >= zlo and zr[e["eid"]][0] <= zhi:
                    bm = np.zeros((Hpx, Wpx), np.uint8)
                    _fill(bm, px(tri[e["eid"]]), 1)
                    total = max(1, int(np.count_nonzero(bm)))
                    for k, n in Counter(smap[(bm > 0) & (smap > 0)].tolist()).items():
                        if n >= min_px:
                            beam_spaces[e["eid"]][sids[k - 1]] = round(n / total, 2)
            # loose items (furniture etc.) -> space by position
            for g in ("furniture", "sanitary", "coverings", "other"):
                for e in grouped.get(g, []):
                    if e.get("storey_eid") == st_eid and not e.get("space_eid") and e.get("pos"):
                        p = px((np.array(e["pos"][:2], np.float64) - origin).astype(np.float32))
                        if 0 <= p[1] < Hpx and 0 <= p[0] < Wpx and smap[p[1], p[0]]:
                            item_space[e["eid"]] = sids[smap[p[1], p[0]] - 1]

    # 3) edges from doors / windows (from either source)
    for d, conn in door_conn.items():
        if len(conn) >= 2:
            add("connected_via", conn[0], conn[1], d, "ifc" if boundaries else "geometry")
    for e in windows:
        wi = eid2id[e["eid"]]
        if wi in window_spaces and (ext.get(e["eid"]) or ext.get(e.get("host_eid")) is True or len(window_spaces[wi]) == 1):
            for s in window_spaces[wi][:1]:
                add("faces_outside", s, None, wi, "ifc" if boundaries else "geometry")
    if boundaries and not shapes:   # adjacency only from shared boundary elements
        for x, ss in wall_spaces.items():
            ss = sorted(ss, key=_idkey)
            for i, a in enumerate(ss):
                for b in ss[i + 1:]:
                    add("adjacent", a, b, x, "ifc")

    # 4) stairs / ramps / elevators connect storeys (z range of their body)
    for g in ("stairs", "ramps", "transport"):
        for e in grouped.get(g, []):
            i, own = eid2id[e["eid"]], sid.get(e.get("storey_eid"))
            hit = [own] if own else []
            if e["eid"] in zr:
                z0, z1 = zr[e["eid"]]
                hit += [s for s, z in storey_elev.items() if z is not None and z0 - 0.3 <= z <= z1 + 0.3]
            hit = sorted(set(hit), key=_idkey)
            if hit:
                stair_storeys[i] = hit
            for a, b in zip(hit, hit[1:]):
                add("connects_storeys", a, b, i)
    return {"relations": relations, "wall_spaces": wall_spaces, "dims": dims, "facade": facade, "ext_geo": ext_geo,
            "beam_spaces": beam_spaces,
            "parts": (space_walls, door_conn, window_spaces, item_space, stair_storeys, sp_geo, st_gross)}


def _fill(target: np.ndarray, tris_px: np.ndarray, value: int) -> None:
    """Paint every triangle on its own. Passing all triangles to one fillPoly call uses the even-odd rule:
    the floor and the ceiling of a closed solid cover the same footprint twice and cancel out."""
    for t in tris_px:
        cv2.fillConvexPoly(target, t, value)


def _tri_area(t: np.ndarray) -> float:
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return float(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum() / 2)


def _cardinal(angle_from_y: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((angle_from_y % 360) + 22.5) // 45) % 8]


# ======================================================================== summary

def _summary(storeys, spaces: Table, tables: dict, types: dict, inventory, area_by_space, b) -> dict:
    s: dict[str, Any] = {"counts": {"storeys": len(storeys), "spaces": len(spaces.rows),
                                    **{g: len(t.rows) for g, t in tables.items()},
                                    **{g: sum(a["count"] for a in v.values()) for g, v in inventory.items()}}}
    sp = spaces.rows
    if sp:
        A, H = SC["net_area"], SC["height"]
        s["spaces"] = {"total_net_area": r(sum(x[A] or 0 for x in sp)),
                       "largest": [[x[0], x[2], x[3], x[A]] for x in sorted(sp, key=lambda x: -(x[A] or 0))[:5]],
                       "tallest": [[x[0], x[2], x[3], x[H]] for x in sorted(sp, key=lambda x: -(x[H] or 0))[:5] if x[H]],
                       "lowest": [[x[0], x[2], x[3], x[H]] for x in sorted(sp, key=lambda x: x[H] or 1e9)[:5] if x[H]],
                       "_columns": ["id", "number", "name", "value"]}
    if "walls" in tables:
        w = tables["walls"].rows
        ext = [x for x in w if x[5] is True]
        inn = [x for x in w if x[5] is False]
        s["walls"] = {"n": len(w), "external": len(ext), "internal": len(inn), "unknown_external": len(w) - len(ext) - len(inn),
                      "load_bearing": sum(x[6] is True for x in w), "non_load_bearing": sum(x[6] is False for x in w),
                      "unknown_load_bearing": sum(x[6] is None for x in w),
                      "length_total": r(sum(x[7] or 0 for x in w)), "length_external": r(sum(x[7] or 0 for x in ext)),
                      "length_internal": r(sum(x[7] or 0 for x in inn)),
                      "net_side_area_external": r(sum(x[10] or 0 for x in ext)), "net_side_area_internal": r(sum(x[10] or 0 for x in inn)),
                      "volume_total": r(sum(x[12] or 0 for x in w), 3)}
    if "doors" in tables:
        d = tables["doors"].rows
        wd = [x for x in d if x[5]]
        s["doors"] = {"n": len(d), "with_width": len(wd), "external": sum(x[8] is True for x in d),
                      "fire_rated": sum(bool(x[7]) for x in d)}
        if wd:
            s["doors"].update(width_min=min(x[5] for x in wd), width_max=max(x[5] for x in wd),
                              width_avg=r(sum(x[5] for x in wd) / len(wd)),
                              widest=[[x[0], x[2], x[5]] for x in sorted(wd, key=lambda x: -x[5])[:5]])
    if "windows" in tables:
        wn = tables["windows"].rows
        ww = [x for x in wn if x[5]]
        s["windows"] = {"n": len(wn), "total_area": r(sum(x[7] or 0 for x in wn)),
                        "by_facade": dict(Counter(x[11] for x in wn if x[11]))}
        if ww:
            s["windows"].update(width_min=min(x[5] for x in ww), width_max=max(x[5] for x in ww),
                                widest=[[x[0], x[2], x[5]] for x in sorted(ww, key=lambda x: -x[5])[:5]])
    if "slabs" in tables:
        by_kind = defaultdict(lambda: {"n": 0, "area": 0.0, "volume": 0.0})
        for x in tables["slabs"].rows:
            k = by_kind[x[5] or "UNKNOWN"]
            k["n"] += 1; k["area"] += x[6] or 0; k["volume"] += x[7] or 0
        s["slabs"] = {k: {"n": v["n"], "area": r(v["area"]), "volume": r(v["volume"], 3)} for k, v in by_kind.items()}
        s["slabs"]["volume_total"] = r(sum(x[7] or 0 for x in tables["slabs"].rows), 3)
    ext_area = s.get("walls", {}).get("net_side_area_external") or 0
    cw = sum(x[8] or 0 for x in tables["curtain_walls"].rows) if "curtain_walls" in tables else 0
    if ext_area or cw:
        s["facade_area"] = {"external_walls_net_side_area": ext_area, "curtain_walls_area": r(cw), "total": r(ext_area + cw)}
    ground = [st for st in storeys if st.elevation is not None and st.elevation >= -0.5]
    if ground:
        g0 = min(ground, key=lambda st: st.elevation)
        s["footprint_area"] = {"storey": g0.id, "gross_floor_area": g0.gross_floor_area}
    s["gross_floor_area_total"] = r(sum(st.gross_floor_area or 0 for st in storeys))
    s["net_floor_area_total"] = r(sum(st.net_floor_area or 0 for st in storeys))
    wall_mats = Counter()
    for t in types.values():
        if t.ifc_class.startswith("IfcWall"):
            for m in t.materials:
                wall_mats[m] += t.count
    if wall_mats:
        s["wall_materials"] = dict(wall_mats.most_common())
    return s
