"""Offline test of polygons, text linking and report (no OCR, no torch needed).

Simulates the layout model output and OCR words for the sanitary area of plan
10498. If torch and the weights are installed, it also runs the real layout
model once as a smoke test.   Run: python tests/test_offline.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from planqa.build_graph import build_graph, parse_area
from planqa.layout import (
    floors_from_segmentation,
    objects_from_segmentation,
    rooms_from_segmentation,
)
from planqa.ocr import OcrWord, group_words, load_image
from planqa.qa import answer, label_categories, route
from planqa.visualize import html_report, layout_blend

img = load_image(ROOT / "data/high_quality_architectural-10498.png")
W, H = img.size

# fake layout model output: (class, x0, y0, x1, y1); 6 = Bath, 7 = Hallway, 11 = Other rooms, 2 = Wall
cls = np.zeros((H, W), np.int64)
for c, x0, y0, x1, y1 in [(7, 539, 258, 614, 308), (6, 539, 43, 595, 166), (6, 599, 43, 673, 118), (11, 539, 170, 673, 258),
                          (6, 618, 315, 673, 391), (7, 512, 308, 614, 391)]:
    cls[y0:y1, x0:x1] = c
cls[43:118, 595:599] = 2
room_prob = np.full((12, H, W), 0.15 / 11, np.float32)
np.put_along_axis(room_prob, cls[None], 0.85, axis=0)
icon = np.zeros((H, W), np.int64)
icon[140:155, 548:560] = 5   # toilet (PE)
icon[100:110, 545:556] = 6   # sink (PE)
icon[60:110, 640:670] = 7    # sauna benches
icon[370:386, 640:652] = 5   # toilet (WC)
icon[200:212, 600:620] = 6   # kitchen sink (KHH)
icon[60:72, 545:575] = 9     # bathtub in PE (so PE has no shower, sauna area none)
for x0 in (300, 360, 420):   # three windows in the outer wall
    icon[45:49, x0:x0 + 30] = 1
for y0 in (180, 240):        # two doors
    icon[y0:y0 + 20, 536:540] = 2
icon_prob = np.full((11, H, W), 0.01, np.float32)
np.put_along_axis(icon_prob, icon[None], 0.9, axis=0)

# fake OCR output at the real label positions ('5,8' + 'm2' are split like real OCR does)
words = [OcrWord(t, list(b), c) for t, b, c in [
    ("PE", (560, 120, 572, 128), .97), ("5,8", (556, 130, 568, 137), .74), ("m2", (570, 130, 580, 137), .91),
    ("S", (608, 67, 614, 74), .93), ("3,5 m2", (600, 75, 620, 82), .90), ("KHH", (562, 188, 576, 196), .95),
    ("8,4 m2", (558, 197, 580, 204), .88), ("WC", (618, 319, 628, 327), .96), ("3,0 m2", (616, 327, 636, 334), .66),
    ("ET", (566, 313, 576, 321), .94), ("9,3 m2", (562, 320, 584, 327), .92), ("19208", (470, 540, 500, 550), .98)]]

graph = build_graph("10498", 0, img.size, group_words(words), rooms_from_segmentation(room_prob, icon_prob),
                    0.8, objects_from_segmentation(icon_prob), floors_from_segmentation(room_prob))
qs = {"How many windows are shown on this floor plan?": 3,
      "Kuinka monta ovea pohjapiirustuksessa on?": 2,
      "How many toilet bowls (WC seats) are drawn on this floor plan?": 2,
      "How many sinks and wash basins (kitchen and bathroom combined) are drawn on this floor plan?": 2,
      "How many showers and bathtubs are drawn on this floor plan in total?": 1,   # PE has a tub; WC is no bathroom
      "How many floors (storeys) does this floor plan show?": 1,
      "How many saunas are on this floor plan?": 1,
      "Kuinka monta kylpy- tai pesuhuonetta (esim. KH, KPH, PH, PESUH, SH) pohjapiirustuksessa on?": 1,
      "How many separate WC rooms (toilet rooms marked WC) are on this floor plan?": 1,
      "How many kitchens are on this floor plan?": 0,
      "How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?": 0,
      "How many balconies (parveke) are on this floor plan?": 0}
answers = []
for q, expected in qs.items():
    a = answer(graph, route(q))
    answers.append({"question": q, "prediction": a.value, "confidence": a.confidence, "fallback": a.fallback,
                    "evidence": a.evidence, "explanation": a.explanation, "answer": expected, "correct": a.value == expected})
    print(f"  {route(q):<17} -> {a.value} (expected {expected})  {a.explanation[:90]}")
out = ROOT / "out/test_offline"
out.mkdir(parents=True, exist_ok=True)
html_report(img, graph, out / "report.html", layout_img=layout_blend(img, room_prob), answers=answers)
assert all(a["correct"] for a in answers), [a["question"] for a in answers if not a["correct"]]

r = {x.label_text: x for x in graph.rooms}
assert parse_area("120 sq ft")[1] == "sqft" and parse_area("8 qm") == (8.0, "m2")
assert len(graph.rooms) == 5  # connecting hallway merges with ET (same class, touching)
assert graph.dropped == {"texts": 2, "rooms": 0, "objects": 0}, graph.dropped  # '5,8 m2' (0.74), '3,0 m2' (0.66)
assert r["PE"].room_type.value == "bathroom" and r["PE"].area_m2 is None  # its area text was below 0.8
assert {"toilet", "sink"} <= {f.value for f in r["PE"].fixtures}
assert r["S"].room_type.value == "sauna" and r["S"].model_class == "Bath" and r["S"].area_m2 == 3.5
assert r["KHH"].area_m2 == 8.4 and r["ET"].room_type.value == "hallway" and r["ET"].area_m2 == 9.3
assert len(graph.floors) == 1 and len(graph.objects) == 11  # 3 windows, 2 doors, 2 toilets, 2 sinks, 1 tub, 1 sauna bench
assert r["WC"].area_m2 is None and "no_area_label" in r["WC"].issues
assert next(t for t in graph.texts if t.text == "19208").room_id is None

# separate thresholds + rebuilding the graph from one detection set (answer.py --sweep)
from planqa.build_graph import thresholds
from planqa.pipeline import graph_from_detections

assert thresholds(0.7) == {"texts": 0.7, "rooms": 0.7, "objects": 0.7}
assert thresholds({"texts": 0.5}) == {"texts": 0.5, "rooms": 0.8, "objects": 0.6}
det = {"source": "10498", "page": 0, "size": img.size, "words": group_words(words),
       "rooms": rooms_from_segmentation(room_prob, icon_prob), "objects": objects_from_segmentation(icon_prob),
       "floors": floors_from_segmentation(room_prob)}
g_low = graph_from_detections(det, {"texts": 0.6})
g_high = graph_from_detections(det, 0.8)
assert g_low.dropped["texts"] == 0 and g_high.dropped["texts"] == 2
assert {x.label_text: x for x in g_low.rooms}["PE"].area_m2 == 5.8      # '5,8 m2' (0.74) now kept
assert {x.label_text: x for x in g_high.rooms}["PE"].area_m2 is None    # detections were not mutated
assert g_low.min_conf == {"texts": 0.6, "rooms": 0.8, "objects": 0.6}

# balconies: unlabelled 'Outdoor' areas only count if the plan labels no outdoor space at all
from planqa.schemas import GraphRoom, RoomType


def outdoor(i, label=None):
    return GraphRoom(id=f"b{i}", room_type=RoomType.other, model_class="Outdoor", type_conf=0.9, type_reason="test",
                     polygon_px=[[0, 0], [1, 0], [1, 1]], box_px=[0, 0, 1, 1], area_px=1, label_text=label)
g = graph.model_copy(deep=True)
g.rooms = [outdoor(1, "PARVEKE"), outdoor(2), outdoor(3), outdoor(4, "TERASSI")]
a = answer(g, "balconies")
assert a.value == 1 and "nicht gezählt" in a.explanation, a
g.rooms = [outdoor(2), outdoor(3)]
g.texts = [t for t in g.texts if not label_categories(t.text) & {"balcony", "terrace"}]
assert answer(g, "balconies").value == 2   # no outdoor labels at all -> layout class decides
print("logic: all checks passed ->", out / "report.html")

# optional: the real layout model
weights = ROOT / "models/model_best_val_loss_var.pkl"
try:
    import torch  # noqa: F401
except ImportError:
    print("layout model: skipped (torch not installed)")
else:
    if weights.is_file():
        from planqa.layout import LayoutModel
        t = time.time()
        model = LayoutModel(str(weights))
        rp, ip = model.segment(img, tta=False)
        rooms = rooms_from_segmentation(rp, ip)
        print(f"layout model: OK, {len(rooms)} rooms in {time.time() - t:.1f}s:",
              ", ".join(f"{x.id} {x.model_class}" for x in rooms))
    else:
        print(f"layout model: skipped ({weights} missing)")
