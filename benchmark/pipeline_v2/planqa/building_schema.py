"""One JSON schema for every source ("building-graph/1"): IFC models and floor-plan images/PDFs.

The agent answers from the same structure no matter where the data came from. A plan simply
fills fewer fields; what is missing is visible (column absent, null, `notes`), so "not in the
data" stays an explicit, explainable outcome.

    source        what was read and how (kind ifc | image | pdf, extraction thresholds, dropped detections)
    building      project / site / building identity (mostly empty for plans)
    storeys  L*   storey list with precomputed aggregates (plans: one entry per drawn floor outline)
    spaces        table, one row per room (IFC IfcSpace / plan room polygon or room label without polygon)
    elements      tables per group: doors, windows, walls, slabs, ..., sanitary, furniture
    annotations   table of texts written on the drawing (plans: OCR) - room labels, areas, dimensions
    types    T*   element types: materials, layers, thickness, profile, Pset_*Common props (IFC only)
    inventory     MEP and other mass elements aggregated per type (IFC only)
    mep           MEP systems, component counts, segment lengths (IFC only)
    schedules     tables read from schedule sheets (door / window / finish schedules), values as written
    legends       code -> meaning (door type marks, material and finish codes, note codes)
    relations     adjacent | connected_via | faces_outside | connects_storeys
    facts         the SAME precomputed answers for every source: counts, areas by category/name/unit
    summary       further precomputed IFC aggregates (walls, doors, windows, slabs, facade ...)
    notes         data gaps, assumptions and how values were derived

Common columns of every table (only present when known; all-null columns are dropped):
    id, name, type (T* id), storey (L* id), space (space id), category (shared vocabulary below),
    conf (detection confidence 0..1, extracted sources only; IFC values are exact as modelled),
    src (where the row comes from: ifc | symbol | room_polygon | room_label | ocr | inferred),
    box_px (plans: [x0, y0, x1, y1] in image pixels) / guid (IFC GlobalId).

Ids: IFC  S* space, W* wall, CW* curtain wall, D* door, WN* window, SL* slab, R* roof, C* column, B* beam,
          M* member, ST* stair, RA* ramp, TR* transport, RL* railing, CV* covering, FT* footing,
          SA* sanitary, FU* furniture.
     plan r* room, t* text, o* symbol (ids of plan_graph.json, so the report can highlight them),
          <room id>.shower for an inferred shower, w* wall run, rx*/ox* detection below threshold (elements.rejected).
'OUTSIDE' = exterior in relations. Units are SI (m, m², m³) unless a schedule keeps values as written.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "building-graph/1"

# ======================================================================== shared vocabulary

# room category -> coarse use (campi `function_category` uses the same coarse values, plus 'outdoor')
SPACE_CATEGORIES: dict[str, str] = {
    "bedroom": "residential",
    "living": "residential",
    "dining": "residential",
    "kitchen": "residential",
    "sauna": "residential",
    "utility": "residential",
    "walk_in_closet": "residential",
    "bathroom": "restroom",
    "wc": "restroom",
    "hallway": "circulation",
    "staircase": "circulation",
    "elevator": "circulation",
    "storage": "storage",
    "technical": "technical",
    "shaft": "technical",
    "office": "office",
    "meeting": "meeting",
    "laboratory": "laboratory",
    "balcony": "outdoor",
    "terrace": "outdoor",
    "garage": "parking",
    "other": "other",
}
OUTDOOR = {"balcony", "terrace"}
SANITARY_CATEGORIES = (
    "toilet",
    "urinal",
    "washbasin",
    "sink",
    "shower",
    "bathtub",
    "bidet",
    "other",
)
FURNITURE_CATEGORIES = (
    "closet",
    "fireplace",
    "sauna_bench",
    "bed",
    "table",
    "seat",
    "kitchen_unit",
    "appliance",
    "chimney",
    "other",
)

# Abbreviations written in plans: one part of a label must match completely ('OH+K' -> living, kitchen)
_ABBREV: list[tuple[str, str]] = [
    ("bathroom", r"KH|KPH|PH|PE|PESUH|SH|KYLPYH\w*"),
    ("wc", r"WC\d?|INVA-?WC|W\.?C\.?"),
    ("sauna", r"S|LH|SAUNA|LOYLYH\w*"),
    ("kitchen", r"K|KK|KT|TK|AK|AVOK\w*|TUPAK\w*|KEITTOK\w*"),
    ("bedroom", r"MH\d*"),
    ("living", r"OH|TUPA"),
    ("dining", r"RUOKAILU\w*"),
    ("hallway", r"ET|AULA|KAYT\w*"),
    ("storage", r"VAR\d?|VARASTO|ULKOVAR\w*"),
    ("walk_in_closet", r"VH"),
    ("utility", r"KHH|KODINH\w*|PESUL\w*"),
    ("technical", r"TEKN\w*|LAMMONJAKO|LJH|IV|IVK"),
    ("balcony", r"P|PARV\w*|LASIT\w*"),
    ("terrace", r"TER|TERASSI"),
    ("garage", r"AT|AUTOTALLI|AUTOSUOJA"),
]
# Words (ASCII-folded, upper case) anywhere in a name; first match wins, several categories allowed
_WORDS: list[tuple[str, str]] = [
    ("sauna", r"SAUNA|SAUN\b|LOYLY"),
    (
        "bathroom",
        r"BATH|BADEZ|\bBAD\b|DUSCH|SHOWER|KYLPY|PESUH|PESURUUM|VANNI|BAGNO|\bBAIN|BADEVAERELSE|WASHROOM|DUSH",
    ),
    ("wc", r"\bWC|TOILET|RESTROOM|LAVATOR|TUALETT|TOALETT|VESSA|GABINETTO|\bLAV\b"),
    (
        "kitchen",
        r"KITCHEN|KUCHE|KUECHE|KEITTIO|KOOK|KOKKEN|CUISINE|CUCINA|PANTRY|TEEKUCHE|PENTRY",
    ),
    (
        "bedroom",
        r"BEDROOM|SCHLAF|MAGAMIS|SOVEVAERELSE|SOVERUM|CHAMBRE|CAMERA DA LETTO|MAKUU|\bBED\b",
    ),
    ("living", r"LIVING|WOHN|OLOHUONE|ELUTUBA|\bSTUE|SEJOUR|SOGGIORNO|LOUNGE"),
    ("dining", r"DINING|ESSEN|RUOKAIL|SOOGI|PRANZO|REPAS|SPISE"),
    ("staircase", r"STAIR|TREPP|TREPI|PORRAS|ESCALIER|SCALA|TRAPPE|STIEGE"),
    ("elevator", r"ELEVATOR|\bLIFT|AUFZUG|HISSI"),
    ("shaft", r"SHAFT|SCHACHT|SHAKT|KUILU|\bVOID\b"),
    (
        "hallway",
        r"CORRID|KORRID|KORIDOR|\bFLUR|DIELE|\bHALL|\bENTR|EINGANG|LOBBY|ESIKODA|ETEINEN|KAYTAVA|COULOIR|"
        r"VESTIB|FOYER|\bGANG\b",
    ),
    ("utility", r"UTILITY|LAUNDRY|WASCHK|HAUSWIRTSCH|KODINHOIT|PESUKOOK|BUANDERIE"),
    ("walk_in_closet", r"WALK-?IN|ANKLEIDE|GARDEROB|VAATEH"),
    (
        "storage",
        r"STORAGE|STORE\b|ABSTELL|REDUIT|LAGER|PANIPAIK|VARASTO|RIPOSTIGLIO|DEPOT|\bKOMERO|CLOSET|"
        r"KELLER|CELLAR|\bBOD\b",
    ),
    (
        "technical",
        r"TECHN|TEKN|ELECTRIC|ELEKTR|SERVER|\bPLANT|MECHANICAL|HVAC|HEIZ|BOILER|KATTILA|\bIT\b|"
        r"VENTILA|SPRINKLER|UTILITIES",
    ),
    (
        "meeting",
        r"MEETING|CONFERENCE|BESPRECH|KONFERENZ|NEUVOTT|NOUPIDAMIS|SITZUNG|BOARDROOM",
    ),
    (
        "office",
        r"OFFICE|BURO|BUERO|KONTOR|TOIMISTO|BUREAU|UFFICIO|WORKSTATION|ARBEITSPLATZ|WORKSPACE",
    ),
    ("laboratory", r"\bLAB"),
    ("balcony", r"BALKON|BALCON|PARVEKE|RODU|LOGGIA|ALTAN"),
    ("terrace", r"TERRA|TERASS|PATIO|VERANDA|DECK\b"),
    ("garage", r"GARAGE|AUTOTALLI|AUTOSUOJA|PARKING|CARPORT|PARKLA|GARAAZ"),
]
# Generic room words: only used when nothing more specific matched ('Zimmer' / 'Pièce' = habitable room)
_GENERIC: list[tuple[str, str]] = [
    (
        "living",
        r"\bZIMMER\b|\bPIECE\b|\bSTANZA\b|\bROOM\b|\bVAERELSE\b|\bHUONE\b|\bTUBA\b",
    )
]
_SANITARY: list[tuple[str, str]] = [  # IFC PredefinedType or name (ASCII-folded, upper)
    ("toilet", r"TOILET|\bWC\b|WATERCLOSET|WATER CLOSET|KLOSETT|TOILETPAN|WC-?ISTUIN"),
    ("urinal", r"URINAL"),
    ("bidet", r"BIDET"),
    ("bathtub", r"BATH\b|BATHTUB|BADEWANNE|WANNE|KYLPYAMME|\bTUB"),
    ("shower", r"SHOWER|DUSCH|SUIHKU"),
    ("sink", r"\bSINK|SPULE|SPUELE|TISKIALL|KITCHENSINK|EVIER"),
    (
        "washbasin",
        r"WASHHANDBASIN|WASHBASIN|BASIN|LAVATORY|LAVABO|WASCHTISCH|WASCHBECKEN|PESUALL|HANDWASH",
    ),
]


def fold(s: Optional[str]) -> str:
    """ASCII, upper case: 'Küche' -> 'KUCHE', 'Kööki' -> 'KOOKI'."""
    if not s:
        return ""
    return (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode()
        .upper()
        .strip()
    )


def space_categories(*names: Optional[str]) -> list[str]:
    """Categories of a room from its label / name / number / object type (multilingual).
    Abbreviation parts first ('OH+K' -> ['living', 'kitchen']), then words; [] if nothing matches."""
    out: list[str] = []
    for n in names:
        f = fold(n)
        if not f:
            continue
        for part in (p for p in re.split(r"[+/,&\s()]+", f.strip(" .:")) if p):
            for cat, rx in _ABBREV:
                if re.fullmatch(rx, part) and cat not in out:
                    out.append(cat)
        if out:
            return out
        for cat, rx in _WORDS:
            if re.search(rx, f) and cat not in out:
                out.append(cat)
        if out:  # 'Wohnen/Essen/Küche' keeps all three
            return out
    for n in names:
        for cat, rx in _GENERIC:
            if re.search(rx, fold(n)):
                return [cat]
    return out


# Full room words for OCR typos ('PARVEXE', 'M.AKUUHVONE', 'ETENEN'): edit distance 1 (2 from 9 letters on)
FUZZY_WORDS = {
    "MAKUUHUONE": "bedroom",
    "OLOHUONE": "living",
    "KEITTIO": "kitchen",
    "KEITTOKOMERO": "kitchen",
    "ETEINEN": "hallway",
    "KYLPYHUONE": "bathroom",
    "PESUHUONE": "bathroom",
    "PARVEKE": "balcony",
    "TERASSI": "terrace",
    "SAUNA": "sauna",
    "LOYLYHUONE": "sauna",
    "VAATEHUONE": "walk_in_closet",
    "VARASTO": "storage",
    "AUTOSUOJA": "garage",
    "AUTOTALLI": "garage",
    "KODINHOITOHUONE": "utility",
    "RUOKAILU": "dining",
    "TYOHUONE": "office",
    "PORRASHUONE": "staircase",
    "TEKNINEN": "technical",
    "SCHLAFZIMMER": "bedroom",
    "WOHNZIMMER": "living",
    "BADEZIMMER": "bathroom",
    "BEDROOM": "bedroom",
    "KITCHEN": "kitchen",
    "BATHROOM": "bathroom",
    "BALCONY": "balcony",
    "TERRACE": "terrace",
}


def _lev(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def fuzzy_categories(text: Optional[str]) -> tuple[list[str], Optional[str]]:
    """Categories of a misspelt label: (categories, the word it was read as) or ([], None).
    Only words of 5+ letters; dots, digits etc. inside a word are ignored ('M.AKUUHVONE')."""
    for tok in re.split(r"[\s+/,&()]+", fold(text)):
        w = re.sub(r"[^A-Z]", "", tok)
        if len(w) < 5:
            continue
        cap = 2 if len(w) >= 9 else 1
        best = min(((_lev(w, k, cap), k) for k in FUZZY_WORDS), default=(cap + 1, None))
        if best[0] <= cap:
            return [FUZZY_WORDS[best[1]]], best[1]
    return [], None


def space_use(categories: Iterable[str]) -> str:
    cats = list(categories)
    return SPACE_CATEGORIES.get(cats[0], "other") if cats else "other"


_FURNITURE: list[tuple[str, str]] = [
    ("bed", r"\bBED|BETT|SANKY|VOODI|\bLIT\b|LETTO"),
    ("closet", r"CLOSET|WARDROBE|SCHRANK|KAAPPI|CABINET|SHELF|REGAL|KAPP"),
    (
        "kitchen_unit",
        r"KITCHEN|KUCHE|KEITTIO|COUNTER|WORKTOP|STOVE|OVEN|FRIDGE|REFRIGERATOR|DISHWASHER",
    ),
    ("table", r"TABLE|TISCH|POYTA|DESK|LAUD"),
    ("seat", r"CHAIR|STUHL|TUOLI|SOFA|COUCH|SEAT|SESSEL|BENCH|TOOL\b"),
    ("fireplace", r"FIREPLACE|KAMIN|TAKKA"),
]


def furniture_category(*names: Optional[str]) -> str:
    for n in names:
        f = fold(n)
        for cat, rx in _FURNITURE:
            if f and re.search(rx, f):
                return cat
    return "other"


def sanitary_category(*names: Optional[str]) -> str:
    for n in names:
        f = fold(n)
        for cat, rx in _SANITARY:
            if f and re.search(rx, f):
                return cat
    return "other"


# ======================================================================== models


class Table(BaseModel):
    """Rows of equal length; row[i] belongs to columns[i]. null = unknown."""

    columns: list[str]
    rows: list[list[Any]] = []


def make_table(
    columns: list[str], rows: list[list[Any]], keep: Iterable[str] = ("id",)
) -> Table:
    """Table without the columns that are null in every row (saves tokens, absence = unknown)."""
    keep = set(keep)
    used = [
        i
        for i, c in enumerate(columns)
        if c in keep or any(r[i] not in (None, [], {}) for r in rows)
    ]
    return Table(
        columns=[columns[i] for i in used], rows=[[r[i] for i in used] for r in rows]
    )


class Extraction(BaseModel):
    method: str  # ifc | ocr+layout | ocr | text_layer
    models: dict[
        str, str
    ] = {}  # e.g. {"ocr": "PP-OCRv6_small", "layout": "CubiCasa5K"}
    thresholds: dict[str, float] = {}  # detections below were dropped
    dropped: dict[str, int] = {}  # how many per kind


class Source(BaseModel):
    file: str
    kind: str = "ifc"  # ifc | image | pdf | dxf
    page: Optional[int] = None
    image_size: Optional[list[int]] = None  # [width, height] px
    ifc_schema: Optional[str] = None  # IFC2X3, IFC4, IFC4X3
    application: Optional[str] = None  # authoring software and version
    view_definition: Optional[str] = None  # e.g. CoordinationView, ReferenceView
    exported_at: Optional[str] = None
    sheet: Optional[str] = None  # drawing sheet number / title (A-801)
    scale: Optional[str] = None  # drawing scale as written (1:100)
    extraction: Optional[Extraction] = None


class Building(BaseModel):
    project: Optional[str] = None
    project_long_name: Optional[str] = None
    site: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None  # ObjectType / OccupancyType
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    georeferenced: bool = False  # IfcMapConversion or site lat/lon present
    true_north_deg: Optional[float] = (
        None  # angle of true north, counter-clockwise from project +Y
    )


class Storey(BaseModel):
    id: str
    name: str
    elevation: Optional[float] = None  # m
    height: Optional[float] = None  # floor-to-floor height to the next storey
    box_px: Optional[list[int]] = None  # plans: outline of the drawn floor
    n_spaces: int = 0
    net_floor_area: float = 0.0  # sum of known space net floor areas (indoor)
    gross_floor_area: Optional[float] = (
        None  # storey quantity or footprint of its floor slabs
    )
    gross_floor_area_src: Optional[str] = (
        None  # model | estimate: ... (then say it is an estimate)
    )
    counts: dict[str, int] = {}  # element group -> count on this storey
    walls: dict[
        str, float
    ] = {}  # length, net_side_area, gross_side_area, volume (sums)
    slabs: dict[str, float] = {}  # volume, area (sums)


class TypeInfo(BaseModel):
    id: str
    ifc_class: str  # class of the elements, e.g. IfcWall
    name: str
    count: int = 0
    predefined: Optional[str] = None
    materials: list[str] = []
    layers: list[
        list[Any]
    ] = []  # [[material, thickness_m], ...] outside -> inside as in IFC
    thickness: Optional[float] = None  # sum of layers or most common element thickness
    profile: Optional[dict[str, Any]] = (
        None  # {"shape": "rectangle", "x": .25, "y": .4} / circle d / I-profile
    )
    props: dict[
        str, Any
    ] = {}  # Pset_*Common values shared by the type (FireRating, IsExternal, ...)


class Relation(BaseModel):
    rel: str  # adjacent | connected_via | faces_outside | connects_storeys
    a: str
    b: Optional[str] = None
    via: Optional[str] = None  # wall / door / window id
    src: str = "geometry"  # geometry | ifc (IfcRelSpaceBoundary)


class BuildingGraph(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    building: Building = Building()
    storeys: list[Storey] = []
    spaces: Table = Field(
        description="id, number, name, storey, categories, use, unit, outdoor, net_area, "
        "gross_area, height, volume, perimeter, walls, contents, conf, src, ..."
    )
    elements: dict[str, Table] = Field(
        default_factory=dict,
        description="doors, windows, walls, ..., sanitary, furniture",
    )
    annotations: Optional[Table] = (
        None  # id, text, kind, conf, space, value, unit, box_px
    )
    types: list[TypeInfo] = []
    inventory: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "MEP and other elements aggregated, not listed: {group: {type_id: {name, count, by_storey, by_space}}}"
        ),
    )
    mep: dict[str, Any] = Field(
        default_factory=dict,
        description="systems with component counts, segment lengths",
    )
    schedules: dict[
        str, Table
    ] = {}  # door_schedule, window_schedule, finish_schedule, ... (as written)
    legends: dict[str, dict[str, str]] = {}  # legend name -> {code: meaning}
    relations: list[Relation] = []
    facts: dict[str, Any] = {}  # see compute_facts
    summary: dict[str, Any] = {}
    notes: list[str] = []  # data gaps and how values were derived


# ======================================================================== facts (same for all sources)

FACT_COUNTS = {  # fact key -> (table, column, value) ; counted rows
    "doors": ("doors", None, None),
    "windows": ("windows", None, None),
    "toilets": ("sanitary", "category", "toilet"),
    "urinals": ("sanitary", "category", "urinal"),
    "washbasins": ("sanitary", "category", "washbasin"),
    "sinks": ("sanitary", "category", "sink"),
    "showers": ("sanitary", "category", "shower"),
    "bathtubs": ("sanitary", "category", "bathtub"),
}
ROOM_FACTS = {
    "bedrooms": "bedroom",
    "kitchens": "kitchen",
    "bathrooms": "bathroom",
    "wc_rooms": "wc",
    "saunas": "sauna",
    "balconies": "balcony",
    "terraces": "terrace",
    "living_rooms": "living",
    "offices": "office",
    "meeting_rooms": "meeting",
    "staircases": "staircase",
    "storage_rooms": "storage",
}


def rows_of(t: Optional[Table]) -> list[dict]:
    return [dict(zip(t.columns, r)) for r in t.rows] if t else []


def compute_facts(g: BuildingGraph, max_names: int = 60) -> dict:
    """Answers to the standard questions, computed the same way for IFC and plans.
    Every count says how it was obtained in facts['how'] (explainability)."""
    spaces = rows_of(g.spaces)
    el = {k: rows_of(v) for k, v in g.elements.items()}
    c: dict[str, Any] = {"storeys": len(g.storeys), "spaces": len(spaces)}
    how: dict[str, str] = {}
    for k, (tab, col, val) in FACT_COUNTS.items():
        rows = el.get(tab)
        if rows is None:
            continue
        sel = [r for r in rows if col is None or r.get(col) == val]
        c[k] = len(sel)
        inferred = sum(r.get("src") == "inferred" for r in sel)
        how[k] = (
            f"rows of elements.{tab}"
            + (f" with {col}={val}" if col else "")
            + (f" ({inferred} inferred, see notes)" if inferred else "")
        )
    if "sinks" in c or "washbasins" in c:
        c["sinks_and_washbasins"] = c.get("sinks", 0) + c.get("washbasins", 0)
    if "showers" in c or "bathtubs" in c:
        c["showers_and_bathtubs"] = c.get("showers", 0) + c.get("bathtubs", 0)
    doors = el.get("doors") or []
    if any(r.get("ext") is not None for r in doors):
        c["doors_external"] = sum(r.get("ext") is True for r in doors)
        c["doors_internal"] = sum(r.get("ext") is False for r in doors)
    for k, cat in ROOM_FACTS.items():
        sel = [s for s in spaces if cat in (s.get("categories") or [])]
        if sel or spaces:
            c[k] = len(sel)
    units = sorted({s["unit"] for s in spaces if s.get("unit")})
    if units:
        c["dwelling_units"] = len(units)
    how["rooms"] = (
        "spaces whose `categories` contain the category (a combined room such as OH+K counts in each)"
    )

    indoor = [s for s in spaces if not s.get("outdoor")]
    a: dict[str, Any] = {}
    known = [s for s in indoor if s.get("net_area") is not None]
    if known:
        a["net_floor_area"] = round(sum(s["net_area"] for s in known), 2)
        a["net_floor_area_spaces_without_area"] = len(indoor) - len(known)
    out = [s for s in spaces if s.get("outdoor") and s.get("net_area") is not None]
    if out:
        a["outdoor_area"] = round(sum(s["net_area"] for s in out), 2)

    def group(key):
        res: dict[str, dict] = {}
        for s in spaces:
            for k in s.get(key) if isinstance(s.get(key), list) else [s.get(key)]:
                if k in (None, ""):
                    continue
                e = res.setdefault(str(k), {"n": 0, "net_area": 0.0, "ids": []})
                e["n"] += 1
                e["ids"].append(s["id"])
                if s.get("net_area") is not None:
                    e["net_area"] += s[
                        "net_area"
                    ]  # unrounded: round once below, not at every step
        for e in res.values():
            e["net_area"] = round(e["net_area"], 2)
            if len(e["ids"]) > 12:
                e["ids"] = e["ids"][:12] + ["..."]
        return res

    a["by_category"] = group("categories")
    by_name = group("name")
    if len(by_name) <= max_names:
        a["by_name"] = by_name
    else:
        a["by_name_note"] = (
            f"{len(by_name)} distinct names: use aggregate(spaces, group_by=name)"
        )
    if units:
        # net area of a dwelling = its indoor spaces (balconies / loggias / terraces separately, as outdoor_area)
        a["by_unit"] = {}
        for u in units:
            mine = [x for x in spaces if x.get("unit") == u]
            inside = [
                x["net_area"]
                for x in mine
                if not x.get("outdoor") and x.get("net_area") is not None
            ]
            outside = [
                x["net_area"]
                for x in mine
                if x.get("outdoor") and x.get("net_area") is not None
            ]
            a["by_unit"][u] = {"n": len(mine), "net_area": round(sum(inside), 2)}
            if outside:
                a["by_unit"][u]["outdoor_area"] = round(sum(outside), 2)
    return {"counts": c, "areas": a, "how": how}
