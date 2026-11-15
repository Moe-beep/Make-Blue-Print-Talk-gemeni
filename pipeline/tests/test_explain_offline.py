"""Offline test of the explanations (planqa/explain.py + explain_view.py) on the synthetic plan of
test_offline.py: elements incl. candidates below threshold, uncertainty maps and hotspots, answers with
evidence / steps / coverage from a simulated agent trace, explain.html.

    python tests/test_explain_offline.py
"""
import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ns = runpy.run_path(str(ROOT / "tests" / "test_offline.py"))   # synthetic plan 10498, its own checks run too

from planqa.explain import build_explanation, write_explanation  # noqa: E402
from planqa.explanation_schema import ExplanationDoc  # noqa: E402
from planqa.pipeline import graph_from_detections  # noqa: E402
from planqa.qa import answer, route  # noqa: E402

img, det, room_prob, icon_prob = ns["img"], ns["det"], ns["room_prob"], ns["icon_prob"]
th = {"texts": 0.8, "objects": 0.6, "rooms": 0.8}
graph = graph_from_detections(det, th)
doors = [o.id for o in graph.objects if o.kind == "door"]
assert doors, "synthetic plan has door symbols"

recs = []
for q in ("How many doors are shown on this floor plan?", "How many bathrooms are on this floor plan?"):
    intent = route(q)
    a = answer(graph, intent)
    recs.append({"question": q, "intent": intent, "prediction": a.value, "confidence": a.confidence, "fallback": a.fallback,
                 "evidence": a.evidence, "explanation": a.explanation, "unit": "count", "answer": a.value, "correct": True})
# the door answer as the LLM agent would produce it (trace with ids, a view, final_answer)
r = recs[0]
r.update(rule_prediction=r["prediction"], rule_evidence=r["evidence"], rule_explanation=r["explanation"],
         rule_confidence=r["confidence"], llm_fallback=False, llm_answer=f"{r['prediction']} Türen",
         llm_reason="Die Tabelle elements.doors enthält alle erkannten Türen. Kontrolle im Bildausschnitt.",
         evidence=["elements.doors"], llm_changes=[], explanation="LLM (gemini-test): ...",
         trace=[{"tool": "overview", "args": {}, "result": "(in the question message)"},
                {"tool": "query_table", "args": {"table": "elements.doors"}, "result": '{"n_matched": %d}' % len(doors),
                 "ids": doors},
                {"tool": "view", "args": {"box_px": [0, 0, 200, 200]},
                 "result": '{"shown": "box", "crop_px": [0, 0, 200, 200], "scale": 1.0}', "ids": []},
                {"tool": "final_answer", "args": {"answer": "x", "reason": "Gezählt aus elements.doors."}}])

doc, heat = build_explanation(img, graph, det, (room_prob, icon_prob), recs, th, {"layout": "cubicasa5k"})
d = json.loads(doc.model_dump_json(exclude_none=True))
ExplanationDoc.model_validate(d)

# elements: kept under graph ids, candidates below threshold as x*
els = d["elements"]
assert all(i in els for i in doors)
assert all(e["status"] == "below_threshold" for i, e in els.items() if i.startswith("x"))
assert any(i.startswith("x") and e["kind"] == "text" for i, e in els.items()), "the two low-confidence OCR words"
assert set(heat) >= {"combined", "rooms", "symbols", "text"} and heat["combined"].mode == "RGBA"
assert 0 <= d["assets"]["uncertain_area"] <= 1

a = d["answers"][0]
counted = [e for e in a["evidence"] if e["role"] == "counted"]
assert sorted(e["element"] for e in counted) == sorted(doors), a["evidence"]
assert [e["order"] for e in counted] == list(range(1, len(doors) + 1))
assert a["references"] == ["elements.doors"] and a["agreement"] == "agree"
kinds = [s["kind"] for s in a["steps"]]
assert kinds == ["route", "read", "read", "tool", "view", "final"], kinds
assert a["steps"][4]["region"] == [0, 0, 200, 200] and a["coverage"]["views"] == [[0, 0, 200, 200]]
assert a["coverage"]["by_kind"]["door"] == [len(doors), len(doors)]
assert all(a["coverage"]["first_seen"][i] == 4 for i in doors)

b = d["answers"][1]   # rules only: no agent coverage, rule step
assert b.get("coverage") is None and b["steps"][-1]["kind"] == "final" and b["agreement"] == "single"

with tempfile.TemporaryDirectory() as tmp:
    write_explanation(doc, heat, img, tmp)
    html = (Path(tmp) / "explain.html").read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html and '"answers"' in html and "/*PAYLOAD*/" not in html
    assert (Path(tmp) / "uncertainty_combined.png").is_file() and (Path(tmp) / "explanation.json").is_file()
print(f"explain: all checks passed ({len(els)} elements, {len(d['hotspots'])} hotspots, "
      f"{len(d['blind_spots'])} blind spots)")
