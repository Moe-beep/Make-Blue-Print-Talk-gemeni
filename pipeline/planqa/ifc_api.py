"""The only module that talks to ifcopenshell: IFC file -> plain raw dicts in SI units.

    pip install ifcopenshell

read_ifc(path, geometry=True) returns
{
  "header":   {file, schema, application, view_definition, exported_at},
  "building": {project, site, name, description, object_type, occupancy, address, lat, lon, georef, true_north_deg},
  "storeys":  [{eid, name, elevation, psets}],
  "elements": [{eid, guid, cls, name, long_name, object_type, predefined, is_mep,
                type: {eid, cls, name, predefined} | None, storey_eid, space_eid, host_eid,
                psets: {pset: {prop: value}}, material: {materials, layers: [[name, m]]},
                pos: [x, y, z], attrs: {...}, profile: {...} | None, flights: [{...}]}],
  "space_boundaries": [[space_eid, element_eid, internal_or_external]],
  "systems":  [{name, kind, members: [eid]}],
  "shapes":   {eid: {"tris": [[[x, y], [x, y], [x, y]], ...], "zmin": z, "zmax": z}}   # metres, world coords
}
Everything numeric is converted to metres / m² / m³ here.
"""
from __future__ import annotations

import os
from pathlib import Path

SKIP = ("IfcOpeningElement", "IfcVirtualElement", "IfcFeatureElement", "IfcAnnotation", "IfcGrid", "IfcPort")
SHAPE_CLASSES = ("IfcSpace", "IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcStair", "IfcRamp",
                 "IfcTransportElement", "IfcSlab", "IfcBeam")
ATTRS_LEN = ("OverallWidth", "OverallHeight", "RiserHeight", "TreadLength")
QTO_LEN = ("Length", "Width", "Height", "Depth", "Perimeter", "Thickness")


def read_ifc(path: str | Path, geometry: bool = True) -> dict:
    import ifcopenshell
    import ifcopenshell.util.element as ue
    import ifcopenshell.util.placement as up
    import ifcopenshell.util.unit as uu

    f = ifcopenshell.open(str(path))
    L = uu.calculate_unit_scale(f)
    try:
        A, V = uu.calculate_unit_scale(f, "AREAUNIT"), uu.calculate_unit_scale(f, "VOLUMEUNIT")
    except TypeError:  # very old ifcopenshell without unit_type
        A, V = L * L, L * L * L

    def qto(psets: dict, qsets: frozenset = frozenset()) -> dict:
        out = {}
        for name, props in psets.items():
            is_q = name in qsets
            name = name or "Unnamed set"   # old exports leave Pset / quantity names empty ($)
            is_q = is_q or name.startswith("Qto_") or "Quantit" in name or name == "BaseQuantities"
            if name in out:   # several unnamed sets: keep all values
                props = {**out[name], **props}
            clean = {}
            for k, v in props.items():
                if k == "id":
                    continue
                if is_q and isinstance(v, (int, float)) and not isinstance(v, bool):
                    v = v * (V if "Volume" in k else A if "Area" in k else L if any(s in k for s in QTO_LEN) else 1)
                clean[k] = v
            out[name] = clean
        return out

    def psets_of(x) -> dict:
        """Property and quantity sets; quantity sets (IfcElementQuantity, whatever their name, e.g. Revit's
        'GSA Space Areas') are converted to SI like Qto_*."""
        try:
            qsets = frozenset(ue.get_psets(x, qtos_only=True))
        except TypeError:   # old ifcopenshell
            qsets = frozenset()
        return qto(ue.get_psets(x), qsets)

    def material(e) -> dict:
        m = ue.get_material(e, should_skip_usage=True)
        if m is None:
            return {}
        if m.is_a("IfcMaterial"):
            return {"materials": [m.Name]}
        if m.is_a("IfcMaterialLayerSet"):
            layers = [[(l.Material.Name if l.Material else None), round(l.LayerThickness * L, 4)] for l in m.MaterialLayers]
            return {"materials": [l[0] for l in layers if l[0]], "layers": layers}
        if m.is_a("IfcMaterialList"):
            return {"materials": [x.Name for x in m.Materials]}
        if m.is_a("IfcMaterialConstituentSet"):
            return {"materials": [c.Material.Name for c in (m.MaterialConstituents or []) if c.Material]}
        if m.is_a("IfcMaterialProfileSet"):
            return {"materials": [p.Material.Name for p in (m.MaterialProfiles or []) if p.Material]}
        return {"materials": [getattr(m, "Name", None)]}

    def profile(e) -> dict | None:
        reps = []
        if getattr(e, "Representation", None):
            reps += e.Representation.Representations
        t = ue.get_type(e)
        for rm in (getattr(t, "RepresentationMaps", None) or []):
            reps.append(rm.MappedRepresentation)
        stack = [i for r in reps for i in r.Items]
        while stack:
            it = stack.pop()
            if it.is_a("IfcMappedItem"):
                stack += it.MappingSource.MappedRepresentation.Items
            elif it.is_a("IfcExtrudedAreaSolid"):
                p, d = it.SweptArea, {"extrusion": round(it.Depth * L, 4)}
                if p.is_a("IfcRectangleProfileDef"):
                    return {**d, "shape": "rectangle", "x": round(p.XDim * L, 4), "y": round(p.YDim * L, 4)}
                if p.is_a("IfcCircleProfileDef"):
                    return {**d, "shape": "circle", "d": round(2 * p.Radius * L, 4)}
                if p.is_a("IfcIShapeProfileDef"):
                    return {**d, "shape": "I", "width": round(p.OverallWidth * L, 4), "depth": round(p.OverallDepth * L, 4)}
                return {**d, "shape": p.is_a().replace("Ifc", "").replace("ProfileDef", "")}
        return None

    def containers(e):
        storey = space = None
        x, seen = e, 0
        while x is not None and seen < 12:
            seen += 1
            if x.is_a("IfcBuildingStorey"):
                storey = x
                break
            if x.is_a("IfcSpace") and x is not e and space is None:
                space = x
            x = ue.get_container(x) or ue.get_aggregate(x)
        return storey, space

    def host(e):
        for rel in getattr(e, "FillsVoids", None) or []:
            for v in rel.RelatingOpeningElement.VoidsElements or []:
                return v.RelatingBuildingElement.id()
        return None

    # ---- header, building
    fn, fd = f.header.file_name, f.header.file_description
    view = next((d for d in (fd.description or []) if "ViewDefinition" in d), None)
    header = {"file": Path(path).name, "schema": f.schema,
              "application": " ".join(x for x in (getattr(fn, "originating_system", ""), getattr(fn, "preprocessor_version", "")) if x) or None,
              "view_definition": view, "exported_at": getattr(fn, "time_stamp", None)}
    proj = (f.by_type("IfcProject") or [None])[0]
    site = (f.by_type("IfcSite") or [None])[0]
    bldg = (f.by_type("IfcBuilding") or [None])[0]

    def dms(v):
        return None if not v else round(v[0] + v[1] / 60 + v[2] / 3600 + (v[3] if len(v) > 3 else 0) / 3.6e9, 6)

    def address(a):
        if not a:
            return None
        parts = list(getattr(a, "AddressLines", None) or []) + [getattr(a, k, None) for k in ("PostalCode", "Town", "Region", "Country")]
        return ", ".join(p for p in parts if p) or None

    north = None
    for ctx in f.by_type("IfcGeometricRepresentationContext"):
        tn = getattr(ctx, "TrueNorth", None)
        if tn is not None:
            import math
            x, y = tn.DirectionRatios[:2]
            north = round(math.degrees(math.atan2(-x, y)), 2)
            break
    bpsets = ue.get_psets(bldg) if bldg else {}
    building = {"project": getattr(proj, "Name", None) or getattr(proj, "LongName", None),
                "project_long_name": getattr(proj, "LongName", None),
                "site": getattr(site, "Name", None), "name": getattr(bldg, "Name", None),
                "description": getattr(bldg, "Description", None), "object_type": getattr(bldg, "ObjectType", None),
                "occupancy": bpsets.get("Pset_BuildingCommon", {}).get("OccupancyType"),
                "address": address(getattr(bldg, "BuildingAddress", None)) or address(getattr(site, "SiteAddress", None)),
                "lat": dms(getattr(site, "RefLatitude", None)), "lon": dms(getattr(site, "RefLongitude", None)),
                "georef": bool(f.schema != "IFC2X3" and f.by_type("IfcMapConversion")), "true_north_deg": north}

    storeys = [{"eid": s.id(), "name": s.Name or s.LongName or f"#{s.id()}",
                "elevation": round((s.Elevation or 0) * L, 4) if s.Elevation is not None else None,
                "psets": psets_of(s)} for s in f.by_type("IfcBuildingStorey")]

    # ---- elements
    elements = []
    for e in list(f.by_type("IfcSpace")) + list(f.by_type("IfcElement")):
        if any(e.is_a(c) for c in SKIP):
            continue
        t = ue.get_type(e)
        st, sp = containers(e)
        try:
            pos = [round(v * L, 3) for v in up.get_local_placement(e.ObjectPlacement)[:3, 3]] if e.ObjectPlacement else None
        except Exception:
            pos = None
        attrs = {}
        for a in ATTRS_LEN + ("NumberOfRisers", "NumberOfRiser", "OperationType", "PartitioningType"):
            v = getattr(e, a, None)
            if v is not None:
                attrs[a] = round(v * L, 4) if a in ATTRS_LEN else v
        rec = {"eid": e.id(), "guid": e.GlobalId, "cls": e.is_a(), "name": e.Name,
               "long_name": getattr(e, "LongName", None), "object_type": getattr(e, "ObjectType", None),
               "predefined": str(getattr(e, "PredefinedType", None) or "") or None,
               "is_mep": e.is_a("IfcDistributionElement") or e.is_a("IfcDistributionFlowElement"),
               "type": {"eid": t.id(), "cls": t.is_a(), "name": t.Name,
                        "predefined": str(getattr(t, "PredefinedType", None) or "") or None} if t else None,
               "storey_eid": st.id() if st else None, "space_eid": sp.id() if sp else None,
               "host_eid": host(e) if e.is_a("IfcDoor") or e.is_a("IfcWindow") else None,
               "psets": psets_of(e), "material": material(e), "pos": pos, "attrs": attrs,
               "profile": profile(e) if any(e.is_a(c) for c in ("IfcColumn", "IfcBeam", "IfcMember", "IfcPile")) else None,
               "flights": []}
        if e.is_a("IfcStair") or e.is_a("IfcRamp"):
            for fl in ue.get_decomposition(e):
                if fl.is_a("IfcStairFlight") or fl.is_a("IfcRampFlight"):
                    ps = ue.get_psets(fl).get("Pset_StairFlightCommon", {})
                    rec["flights"].append({
                        "risers": getattr(fl, "NumberOfRisers", None) or getattr(fl, "NumberOfRiser", None) or ps.get("NumberOfRiser"),
                        "riser_height": round(fl.RiserHeight * L, 4) if getattr(fl, "RiserHeight", None) else ps.get("RiserHeight"),
                        "tread_length": round(fl.TreadLength * L, 4) if getattr(fl, "TreadLength", None) else ps.get("TreadLength")})
        elements.append(rec)

    boundaries = []
    for rel in f.by_type("IfcRelSpaceBoundary"):
        sp, el = rel.RelatingSpace, rel.RelatedBuildingElement
        if sp is not None and el is not None and sp.is_a("IfcSpace"):
            boundaries.append([sp.id(), el.id(), str(getattr(rel, "InternalOrExternalBoundary", "") or "") or None])

    systems = []
    for s in f.by_type("IfcSystem"):
        if s.is_a("IfcZone") or s.is_a("IfcStructuralAnalysisModel") or s.is_a("IfcBuildingSystem"):
            continue
        members = [o.id() for g in (s.IsGroupedBy or []) for o in (g.RelatedObjects or [])]
        systems.append({"name": s.Name, "kind": str(getattr(s, "PredefinedType", None) or s.ObjectType or "") or None,
                        "members": members})

    shapes = _shapes(f, [e for c in SHAPE_CLASSES for e in f.by_type(c)]) if geometry else {}
    return {"header": header, "building": building, "storeys": storeys, "elements": elements,
            "space_boundaries": boundaries, "systems": systems, "shapes": shapes}


def _shapes(f, products) -> dict:
    """Triangulated bodies in world coordinates (metres), reduced to 2D triangles + z range."""
    import numpy as np
    import ifcopenshell.geom

    s = ifcopenshell.geom.settings()
    try:
        s.set("use-world-coords", True)       # ifcopenshell >= 0.8
    except Exception:
        s.set(s.USE_WORLD_COORDS, True)       # 0.7
    out = {}
    it = ifcopenshell.geom.iterator(s, f, max(1, (os.cpu_count() or 2) - 1), include=products)
    if not it.initialize():
        return out
    while True:
        sh = it.get()
        v = np.asarray(sh.geometry.verts, float).reshape(-1, 3)
        fc = np.asarray(sh.geometry.faces, int).reshape(-1, 3)
        if len(v) and len(fc):
            out[sh.id] = {"tris": np.round(v[fc][:, :, :2], 3).tolist(),
                          "zmin": round(float(v[:, 2].min()), 3), "zmax": round(float(v[:, 2].max()), 3)}
        if not it.next():
            break
    return out
