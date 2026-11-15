"""Raw output of each model -> building-graph/1, the format the rules (planqa/campi_map.py) read.

One function per model. Nothing the models give is dropped: data without a rule yet (polygons, pixel
areas, wall segments and junctions, untyped room boxes, detections below the threshold) is kept in
extra columns / tables, so a later rule can use it without rerunning the model.

Counts a model cannot observe are removed from facts.counts after compute_facts (a model without a
shower class would otherwise report 0 showers); facts.not_measured lists them.
Coordinates: box_px / polygon_px are pixels of the original image (model canvases are mapped back);
for PDFs they are PDF points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipelines.raster2seq_profiles import (
    DOOR_LABEL,
    R2G_LABELS,
    WINDOW_LABEL,
    Raster2SeqProfile,
)
from planqa.building_schema import (
    OUTDOOR,
    ROOM_FACTS,
    BuildingGraph,
    Extraction,
    Source,
    Storey,
    compute_facts,
    fuzzy_categories,
    make_table,
    space_categories,
    space_use,
)

# Raster-to-Graph/util/semantics_dict.py room classes (wall/outside are dropped by its room extraction)
RASTER2GRAPH_LABELS = tuple(sorted(set(R2G_LABELS.values()) - {"outside"}))
# same symbol threshold as planqa/build_graph.py DEFAULT_MIN_CONF["objects"]
LAYOUT_MIN_CONF = 0.6
# 2DPlan2Strct model card default threshold
PLAN2STRCT_MIN_SCORE = 0.35

# OCR words: reading confidence a room label needs (planqa/build_graph.py DEFAULT_MIN_CONF["texts"])
TEXT_OCR_MIN_CONF = 0.6
TEXT_COLS = ["id", "text", "kind", "conf", "det_conf", "line", "space", "box_px", "polygon_px"]
ROOM_LABEL = "room_label"

SPACE_COLS = [
    "id",
    "number",
    "name",
    "storey",
    "categories",
    "use",
    "unit",
    "outdoor",
    "net_area",
    "conf",
    "src",
    "model_class",
    "issues",
    "box_px",
    "polygon_px",
    "area_px",
    "void_area",
]
OBJ_COLS = [
    "id",
    "storey",
    "space",
    "category",
    "conf",
    "src",
    "box_px",
    "polygon_px",
    "area_px",
]
WALL_COLS = [
    "id",
    "storey",
    "category",
    "length_px",
    "conf",
    "src",
    "box_px",
    "segment_px",
]
JUNCTION_COLS = ["id", "point_px", "src"]
KEEP_SPACE = ("id", "name", "storey", "categories", "net_area")
KEEP_OBJ = ("id", "storey", "category", "conf")

PDF_NOTE = "areas from the vector room polygons and the printed scale; outdoor rooms by label, shafts/voids by fill colour"
STOREY = "L0"  # one drawing = one storey unless the model separates storey outlines
# counts every source reports, whatever its classes
BASE_COUNTS = {"spaces"}
# layout symbol kind -> (element group, category)
LAYOUT_SYMBOLS = {
    "door": ("doors", None),
    "window": ("windows", None),
    "toilet": ("sanitary", "toilet"),
    "sink": ("sanitary", "sink"),
    "bathtub": ("sanitary", "bathtub"),
    "closet": ("furniture", "closet"),
    "sauna_bench": ("furniture", "sauna_bench"),
    "fireplace": ("furniture", "fireplace"),
}
# room types the layout model gives (planqa/layout.py ROOM_TO_TYPE values, plus its derived sauna)
LAYOUT_ROOM_NAMES = ("balcony", "kitchen", "living_room", "bedroom", "bathroom", "hallway", "storage", "garage",
                     "other", "sauna")


# ------------------------------------------------------------------------------------------- helpers


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace area of a polygon given as [[x, y], ...]."""
    if len(points) < 3:
        return 0.0
    xy = np.asarray(points, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def letterbox_to_image(
    points: list[list[float]], canvas: int, width: int, height: int
) -> list[list[float]]:
    """Canvas coordinates of a centred, aspect-preserving resize back to original pixels."""
    scale = min(canvas / height, canvas / width)
    left = (canvas - int(width * scale)) // 2
    top = (canvas - int(height * scale)) // 2
    return [
        [round((x - left) / scale, 1), round((y - top) / scale, 1)] for x, y in points
    ]


def bbox(points: list[list[float]]) -> list[int]:
    xy = np.asarray(points, float)
    return [
        int(xy[:, 0].min()),
        int(xy[:, 1].min()),
        int(xy[:, 0].max()),
        int(xy[:, 1].max()),
    ]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def space_row(
    i: int,
    name: str | None,
    src: str,
    *,
    polygon=None,
    box=None,
    conf=None,
    model_class=None,
    number=None,
    unit=None,
    net_area=None,
    outdoor=None,
    issues=None,
    void_area=None,
    area_px=None,
    categories=None,
) -> list:
    if categories is None:
        categories = space_categories(name) if name else []
    if outdoor is None:
        outdoor = bool(OUTDOOR & set(categories))
    if polygon is not None and area_px is None:
        area_px = round(polygon_area(polygon), 1)
    return [
        f"r{i:02d}",
        number,
        name,
        STOREY,
        categories,
        space_use(categories),
        unit,
        outdoor,
        net_area,
        conf,
        src,
        model_class,
        issues or [],
        box or (bbox(polygon) if polygon else None),
        polygon,
        area_px,
        void_area,
    ]


def obj_row(
    i: int,
    category: str | None,
    src: str,
    *,
    box=None,
    polygon=None,
    conf=None,
    area_px=None,
) -> list:
    return [
        f"o{i:03d}",
        STOREY,
        None,
        category,
        conf,
        src,
        box or (bbox(polygon) if polygon else None),
        polygon,
        area_px,
    ]


def measured_room_facts(names: tuple[str, ...]) -> set[str]:
    """Room counts a label set can observe: a model without a 'balcony' class cannot count balconies."""
    categories = {c for name in names for c in space_categories(name)}
    return {fact for fact, cat in ROOM_FACTS.items() if cat in categories}


def finish(g: BuildingGraph, measured: set[str]) -> BuildingGraph:
    """compute_facts, then remove the counts this source cannot observe."""
    facts = compute_facts(g)
    counts = facts["counts"]
    not_measured = sorted(k for k in counts if k not in measured | BASE_COUNTS)
    facts["counts"] = {k: v for k, v in counts.items() if k in measured | BASE_COUNTS}
    facts["not_measured"] = not_measured
    g.facts = facts
    return g


def new_graph(
    file: str, kind: str, size, method: str, models: dict, **extraction
) -> BuildingGraph:
    return BuildingGraph(
        source=Source(
            file=file,
            kind=kind,
            image_size=list(size) if size else None,
            extraction=Extraction(method=method, models=models, **extraction),
        ),
        storeys=[Storey(id=STOREY, name="drawn floor")],
        spaces=make_table(SPACE_COLS, [], KEEP_SPACE),
    )


def opening_tables(rows_by_group: dict[str, list[list]]) -> dict:
    return {
        g: make_table(OBJ_COLS, rows, KEEP_OBJ) for g, rows in rows_by_group.items()
    }


# ------------------------------------------------------------------------------------------- models


def raster2seq_graph(
    output_dir: Path, stem: str, image: Path, profile: Raster2SeqProfile
) -> BuildingGraph | None:
    paths = sorted(output_dir.glob(f"*/jsons/{stem}.json"))
    if not paths:
        return None
    width, height = image_size(image)
    spaces, doors, windows = [], [], []
    for instance in json.loads(paths[0].read_text(encoding="utf-8")):
        label = profile.labels.get(
            instance["category_id"], f"class_{instance['category_id']}"
        )
        canvas_poly = (
            np.asarray(instance["segmentation"], float).reshape(-1, 2).tolist()
        )
        poly = letterbox_to_image(canvas_poly, profile.canvas_size, width, height)
        if label in (DOOR_LABEL, WINDOW_LABEL):
            target = doors if label == DOOR_LABEL else windows
            target.append(
                obj_row(
                    len(doors) + len(windows) + 1,
                    None,
                    "raster2seq",
                    polygon=poly,
                    area_px=round(polygon_area(poly), 1),
                )
            )
        else:
            spaces.append(
                space_row(
                    len(spaces) + 1,
                    label,
                    "raster2seq",
                    polygon=poly,
                    model_class=label,
                )
            )
    g = new_graph(
        image.name,
        "image",
        (width, height),
        "raster2seq",
        {"raster2seq": profile.checkpoint},
        thresholds={},
        dropped={},
    )
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    measured = {"spaces"} | measured_room_facts(tuple(profile.labels.values()))
    if profile.has_openings:
        g.elements = opening_tables({"doors": doors, "windows": windows})
        measured |= {"doors", "windows"}
    return finish(g, measured)


def raster2graph_graph(
    output_dir: Path, stem: str, image: Path, labels: tuple[str, ...]
) -> BuildingGraph | None:
    path = output_dir / "json" / f"{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = image_size(image)
    canvas = data["canvas_size"]
    to_px = lambda pts: letterbox_to_image(pts, canvas, width, height)
    spaces = [
        space_row(
            i,
            r["semantic"],
            "raster2graph",
            polygon=to_px(r["polygon"]),
            model_class=r["semantic"],
        )
        for i, r in enumerate(data["rooms"], 1)
    ]
    walls = []
    for i, (a, b) in enumerate(data["wall_segments"], 1):
        (ax, ay), (bx, by) = to_px([a, b])
        walls.append(
            [
                f"w{i:03d}",
                STOREY,
                "wall",
                round(float(np.hypot(bx - ax, by - ay)), 1),
                None,
                "raster2graph",
                bbox([[ax, ay], [bx, by]]),
                [[ax, ay], [bx, by]],
            ]
        )
    junctions = [
        [f"j{i:03d}", p, "raster2graph"]
        for i, p in enumerate(to_px(data["junctions"]), 1)
    ]
    g = new_graph(
        image.name,
        "image",
        (width, height),
        "raster2graph",
        {"raster2graph": "Raster-to-Graph"},
    )
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    g.elements = {
        "walls": make_table(WALL_COLS, walls, ("id", "category", "length_px")),
        "wall_junctions": make_table(JUNCTION_COLS, junctions, ("id", "point_px")),
    }
    return finish(g, {"spaces"} | measured_room_facts(labels))


def layout_graph(output_dir: Path, stem: str, image: Path) -> BuildingGraph | None:
    path = output_dir / f"{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list] = {
        "doors": [],
        "windows": [],
        "sanitary": [],
        "furniture": [],
    }
    rejected = []
    for i, o in enumerate(data["objects"], 1):
        group, category = LAYOUT_SYMBOLS.get(o["kind"], ("furniture", o["kind"]))
        row = obj_row(
            i, category, "layout", box=o["box_px"], conf=o["conf"], area_px=o["area_px"]
        )
        if o["conf"] < LAYOUT_MIN_CONF:
            rejected.append([*row[:3], o["kind"], *row[4:]])
        else:
            groups[group].append(row)
    rooms = data.get("rooms", [])
    spaces = [
        space_row(
            i,
            r["room_type"],
            "layout",
            polygon=r["polygon_px"],
            box=r["box_px"],
            conf=r["type_conf"],
            model_class=r["model_class"],
            area_px=r["area_px"],
        )
        for i, r in enumerate(rooms, 1)
    ]
    g = new_graph(
        image.name,
        "image",
        data["image_size_px"],
        "layout",
        {"layout": "CubiCasa5K"},
        thresholds={"objects": LAYOUT_MIN_CONF},
        dropped={"objects": len(rejected)},
    )
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    floors = data.get("floors")
    if floors:
        g.storeys = [
            Storey(id=f"L{i}", name=f"drawn floor {i + 1}", box_px=b)
            for i, b in enumerate(floors)
        ]
    g.elements = opening_tables(groups)
    if rejected:
        g.elements["rejected"] = make_table(
            OBJ_COLS, rejected, ("id", "category", "conf")
        )
    # no shower class, and sinks are not split into kitchen sink / washbasin without the room logic of planqa
    measured = {"doors", "windows", "toilets", "bathtubs", "sinks_and_washbasins"}
    if "rooms" in data:
        measured |= {"spaces"} | measured_room_facts(LAYOUT_ROOM_NAMES)
    if floors:
        measured.add("storeys")
    return finish(g, measured)


def plan2strct_graph(output_dir: Path, stem: str, image: Path) -> BuildingGraph | None:
    path = output_dir / f"{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    doors, windows, walls, spaces, rejected = [], [], [], [], []
    for i, d in enumerate(data["detections"], 1):
        box = [round(v, 1) for v in d["box_xyxy"]]
        score = round(d["score"], 3)
        if d["score"] < PLAN2STRCT_MIN_SCORE:
            rejected.append(
                obj_row(
                    i,
                    d["label"],
                    "plan2strct",
                    box=box,
                    conf=score,
                    area_px=round(box_area(box), 1),
                )
            )
        elif d["label"] == "door":
            doors.append(
                obj_row(
                    i,
                    None,
                    "plan2strct",
                    box=box,
                    conf=score,
                    area_px=round(box_area(box), 1),
                )
            )
        elif d["label"] == "window":
            windows.append(
                obj_row(
                    i,
                    None,
                    "plan2strct",
                    box=box,
                    conf=score,
                    area_px=round(box_area(box), 1),
                )
            )
        elif d["label"] == "wall":
            length = max(box[2] - box[0], box[3] - box[1])
            walls.append(
                [
                    f"w{i:03d}",
                    STOREY,
                    "wall",
                    round(length, 1),
                    score,
                    "plan2strct",
                    box,
                    None,
                ]
            )
        else:  # room: a box without type
            spaces.append(
                space_row(
                    len(spaces) + 1,
                    None,
                    "plan2strct",
                    box=box,
                    conf=score,
                    area_px=round(box_area(box), 1),
                    issues=["untyped_room_box"],
                )
            )
    g = new_graph(
        image.name,
        "image",
        (data["width"], data["height"]),
        "plan2strct",
        {"plan2strct": "2DPlan2Strct RF-DETR Medium"},
        thresholds={"objects": PLAN2STRCT_MIN_SCORE},
        dropped={"objects": len(rejected)},
    )
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    g.elements = {
        **opening_tables({"doors": doors, "windows": windows}),
        "walls": make_table(WALL_COLS, walls, ("id", "category", "conf")),
    }
    if rejected:
        g.elements["rejected"] = make_table(
            OBJ_COLS, rejected, ("id", "category", "conf")
        )
    return finish(g, {"spaces", "doors", "windows"})


def pdf_plan_graph(plan: dict, pdf: Path) -> BuildingGraph:
    """Vector PDF geometry (pipelines/pdf_plan.py): rooms with number, label, apartment and exact area."""
    spaces = []
    for i, (number, r) in enumerate(sorted(plan["rooms"].items()), 1):
        excluded = r["excluded_from_net_area"]
        void = excluded == "shaft_or_void"
        spaces.append(
            space_row(
                i,
                r["label"],
                "pdf_geometry",
                number=number,
                unit=r["apartment"],
                polygon=r.get("polygon_pt"),
                area_px=r.get("area_pt2"),
                net_area=None if void else r.get("area_m2_precise", r["net_floor_area_m2"]),
                outdoor=excluded == "outdoor",
                void_area=r["net_floor_area_m2"] if void else None,
                issues=["shaft_or_void: not part of the net floor area"]
                if void
                else [],
            )
        )
    g = new_graph(
        pdf.name,
        "pdf",
        None,
        "text_layer+vector_geometry",
        {"pdf_plan": "pipelines/pdf_plan.py"},
    )
    g.source.scale = f"1:{plan['scale_denominator']}"
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE + ("number", "unit"))
    g.notes = [PDF_NOTE]
    measured = {"storeys", "spaces"} | set(ROOM_FACTS)
    if plan["apartments"]:
        measured.add("dwelling_units")
    g = finish(g, measured)
    if plan["apartments"]:
        # an apartment outline can hold no numbered room: count the markers drawn, not the units of rooms
        g.facts["counts"]["dwelling_units"] = len(plan["apartments"])
        g.facts["how"]["dwelling_units"] = "apartment markers (App. A, Whg. B) with an outline on the plan"
    return g


# ------------------------------------------------------------------------------------------- YOLO26-Text + PaddleOCR

def label_categories(text: str) -> list[str]:
    """Room categories a word names: exact vocabulary (MH, KH, WC, S, OH+K, ...), else a misspelt long word."""
    return space_categories(text) or fuzzy_categories(text)[0]


def text_ocr_graph(output_dir: Path, stem: str, image: Path) -> BuildingGraph | None:
    """Words read by YOLO26-Text + PP-OCRv6 (pipelines/text_ocr_infer.py).

    Every word goes to annotations. A word that names a room type and is read with confidence >=
    TEXT_OCR_MIN_CONF is one room (its box is the label's box: the room outline is not known), as in the
    plan pipeline of planqa: one label = one room, a combined label (OH+K) counts in each category.
    """
    path = output_dir / f"{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations, spaces = [], []
    for i, t in enumerate(data["texts"], 1):
        polygon = [[round(x, 1), round(y, 1)] for x, y in t["polygon"]]
        categories = label_categories(t["text"])
        is_room = bool(categories) and t["confidence"] >= TEXT_OCR_MIN_CONF
        space_id = None
        if is_room:
            spaces.append(space_row(len(spaces) + 1, t["text"], "text_ocr", box=bbox(polygon),
                                    conf=round(t["confidence"], 3), categories=categories,
                                    issues=["label_only: box is the label, not the room outline"]))
            space_id = spaces[-1][0]
        annotations.append([f"t{i:03d}", t["text"], ROOM_LABEL if is_room else "text", round(t["confidence"], 3),
                            round(t["detection_confidence"], 3), t["line"], space_id, bbox(polygon), polygon])
    g = new_graph(image.name, "image", (data["width"], data["height"]), "ocr", {"text_ocr": data["model"]},
                  thresholds={"texts": TEXT_OCR_MIN_CONF})
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    g.annotations = make_table(TEXT_COLS, annotations, ("id", "text", "kind", "conf"))
    # every word was read: a room type without a label on the plan really is absent
    return finish(g, {"spaces"} | set(ROOM_FACTS))


# ------------------------------------------------------------------------------------------- YOLO (ONNX)

# YOLO class -> (element group, category); doorways are openings without a leaf: not doors (schema)
YOLO_ELEMENTS = {
    "window": ("windows", None), "door": ("doors", None), "doorway": ("doorways", None),
    "toilet": ("sanitary", "toilet"), "sink": ("sanitary", "sink"),
    "shower": ("sanitary", "shower"), "bathtub": ("sanitary", "bathtub"),
}
# YOLO room class -> space name (its categories come from the building-graph vocabulary)
YOLO_ROOMS = {"sauna": "sauna", "kitchen": "kitchen", "bedroom": "bedroom", "wc_room": "wc",
              "bathroom": "bathroom", "balcony": "balcony"}
YOLO_FLOOR = "floor"
YOLO_COUNTS = {"door": {"doors"}, "window": {"windows"}, "toilet": {"toilets"}, "sink": {"sinks_and_washbasins"},
               "shower": {"showers"}, "bathtub": {"bathtubs"}}


def yolo_thresholds(spec: dict) -> dict[str, float]:
    """Per-class thresholds tuned for the ONNX path when the spec has them, else the PyTorch-path ones."""
    return spec.get("thresholds_onnx_path") or spec["thresholds"]


def yolo_graph(output_dir: Path, stem: str, image: Path, spec: dict, source: str) -> BuildingGraph | None:
    """Boxes of a YOLO floor-plan detector (pipelines/yolo_infer.py) with its spec's per-class thresholds."""
    path = output_dir / f"{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds = yolo_thresholds(spec)
    groups: dict[str, list] = {}
    spaces, floors, rejected = [], [], []
    for i, d in enumerate(data["detections"], 1):
        label, score, box = d["label"], d["score"], d["box_xyxy"]
        area = round(box_area(box), 1)
        if score < thresholds.get(label, 1.0):
            rejected.append(obj_row(i, label, source, box=box, conf=score, area_px=area))
        elif label in YOLO_ELEMENTS:
            group, category = YOLO_ELEMENTS[label]
            groups.setdefault(group, []).append(obj_row(i, category, source, box=box, conf=score, area_px=area))
        elif label in YOLO_ROOMS:
            spaces.append(space_row(len(spaces) + 1, YOLO_ROOMS[label], source, box=box, conf=score,
                                    model_class=label, area_px=area))
        elif label == YOLO_FLOOR:
            floors.append([int(v) for v in box])
    g = new_graph(image.name, "image", (data["width"], data["height"]), "yolo", {source: data["model"]},
                  thresholds={f"class:{k}": v for k, v in thresholds.items()}, dropped={"objects": len(rejected)})
    g.spaces = make_table(SPACE_COLS, spaces, KEEP_SPACE)
    g.elements = {group: make_table(OBJ_COLS, rows, KEEP_OBJ) for group, rows in groups.items()}
    for group in {YOLO_ELEMENTS[c][0] for c in spec["classes"] if c in YOLO_ELEMENTS}:
        g.elements.setdefault(group, make_table(OBJ_COLS, [], KEEP_OBJ))  # class known, none found: count 0
    if rejected:
        g.elements["rejected"] = make_table(OBJ_COLS, rejected, ("id", "category", "conf"))
    if floors:
        g.storeys = [Storey(id=f"L{i}", name=f"drawn floor {i + 1}", box_px=b) for i, b in enumerate(floors)]
    classes = set(spec["classes"])
    measured = set().union(*(YOLO_COUNTS[c] for c in classes if c in YOLO_COUNTS))
    if {"shower", "bathtub"} <= classes:
        measured.add("showers_and_bathtubs")
    room_names = tuple(YOLO_ROOMS[c] for c in classes if c in YOLO_ROOMS)
    if room_names:
        measured |= {"spaces"} | measured_room_facts(room_names)
    if YOLO_FLOOR in classes:
        measured.add("storeys")
    return finish(g, measured)
