"""Offline test of the frontend service (planqa/service_core.py + serve.py): synthetic plan of test_offline.py
through a fake pipeline (no OCR / torch), synthetic IFC of test_ifc_offline.py with a scripted agent, and the
HTTP endpoints on a random local port.

    python tests/test_service_offline.py
"""
import base64
import io
import json
import runpy
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
ns = runpy.run_path(str(ROOT / "tests" / "test_offline.py"))   # synthetic plan 10498 (its own checks run too)

from planqa.agent import Agent, ScriptedBackend  # noqa: E402
from planqa.explanation_schema import Answer  # noqa: E402
from planqa.pipeline import graph_from_detections  # noqa: E402
from planqa.service_core import Doc, Store  # noqa: E402
from test_ifc_offline import model  # noqa: E402

img, det, room_prob, icon_prob = ns["img"], ns["det"], ns["room_prob"], ns["icon_prob"]
TH = {"texts": 0.8, "objects": 0.6, "rooms": 0.8}


class FakePipeline:
    """Same interface as planqa.pipeline.Pipeline, returns the synthetic detections."""

    def process(self, path, out_dir, page=0, report=True):
        self.last_image, self.last_detections, self.last_probs = img, det, (room_prob, icon_prob)
        self.last_layout_img = None
        return graph_from_detections(det, TH), {}


def png_bytes():
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


with tempfile.TemporaryDirectory() as tmp:
    # ------------------------------------------------------------------ plan, rules only
    store = Store(Path(tmp), agent=None, pipeline_factory=FakePipeline, min_conf=TH, models={"layout": "fake"})
    doc = store.submit("plan.png", png_bytes(), background=False)
    assert doc.status == "ready", doc.error
    info = doc.info()
    assert info["explanation"]["source"]["image_size"] == list(img.size)
    assert "answers" not in info["explanation"] and info["assets"]["combined"].endswith("uncertainty_combined.png")
    assert store.submit("plan.png", png_bytes()) is doc, "same bytes + page -> same document"

    r = store.ask(doc.id, "How many doors are shown on this floor plan?")
    a = r["answer"]
    Answer.model_validate(a)
    doors = [o.id for o in doc.graph.objects if o.kind == "door"]
    counted = [e["element"] for e in a["evidence"] if e["role"] == "counted"]
    assert a["value"] == len(doors) and sorted(counted) == sorted(doors), a
    assert all(i in r["elements"] and r["elements"][i]["geometry"]["bbox"] for i in counted)
    assert r["image_size"] == list(img.size) and a["id"] == "q1" and a["unit"] == "count"
    assert a["agreement"] == "single" and a["steps"][-1]["kind"] == "final"

    r2 = store.ask(doc.id, "What is the colour of the facade?")   # unknown question type, no agent
    assert r2["answer"]["abstained"] and "LLM" in r2["answer"]["abstain_reason"] and r2["answer"]["id"] == "q2"

    # ------------------------------------------------------------------ IFC with a scripted agent
    script = [[("query_table", {"table": "elements.doors"})],
              [("final_answer", {"answer": "Es gibt 2 Türen.", "value": 2, "unit": "count", "evidence": ["D1", "D2"],
                                 "confidence": 0.9, "reason": "elements.doors hat 2 Zeilen. Beide auf L0.",
                                 "provenance": "stated"})]]
    ifc_store = Store(Path(tmp) / "ifc", agent=Agent(ScriptedBackend(script)))
    raw = model()
    idoc = Doc(id="ifc1", name="m.ifc", kind="ifc", page=0, path=Path(tmp) / "m.ifc", out_dir=Path(tmp) / "ifc1")
    idoc.out_dir.mkdir()
    ifc_store.load_ifc_raw(idoc, raw)
    idoc.status = "ready"
    ifc_store.docs[idoc.id] = idoc
    assert idoc.index["D1"]["kind"] == "door" and idoc.viewer_ids["D1"].startswith("#")
    eid_of = {e["guid"]: e["eid"] for e in raw["elements"]}
    assert idoc.viewer_ids["D1"] == f"#{eid_of[idoc.index['D1']['guid']]}"
    ri = ifc_store.ask("ifc1", "Wie viele Türen gibt es?")
    ai = ri["answer"]
    Answer.model_validate(ai)
    assert ai["value"] == 2 and [e["element"] for e in ai["evidence"]] == ["D1", "D2"]
    assert all(e["role"] == "counted" for e in ai["evidence"]) and ai["sources"]["agent"]["model"] == "scripted"
    assert set(ri["viewer_ids"]) >= {"D1", "D2"} and ri["elements"]["D1"]["geometry"]["guid"]
    assert [s["kind"] for s in ai["steps"]][-1] == "final" and "Provenance according to the agent: stated" in ai["limitations"]
    assert ri["image_size"] is None and idoc.info()["summary"]["elements"]["doors"] == 2

    # IFC without agent -> clear error
    try:
        noagent = Store(Path(tmp) / "x")
        noagent.docs["ifc1"] = idoc
        noagent.ask("ifc1", "x?")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e)

    # ------------------------------------------------------------------ HTTP
    import serve
    from http.server import ThreadingHTTPServer
    http_store = Store(Path(tmp) / "http", agent=None, pipeline_factory=FakePipeline, min_conf=TH)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(http_store, {"agent": None}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type")

    st, body, _ = call("GET", "/health")
    assert st == 200 and json.loads(body)["ok"]
    st, body, _ = call("POST", "/documents", {"name": "p.png", "dataBase64": base64.b64encode(png_bytes()).decode()})
    assert st == 202, body
    did = json.loads(body)["id"]
    for _ in range(100):
        st, body, _ = call("GET", f"/documents/{did}")
        if json.loads(body)["status"] in ("ready", "error"):
            break
        time.sleep(0.05)
    info = json.loads(body)
    assert info["status"] == "ready", info
    st, body, ctype = call("GET", info["assets"]["combined"])
    assert st == 200 and ctype == "image/png" and body[:4] == b"\x89PNG"
    st, body, _ = call("GET", f"/documents/{did}/assets/../../etc")
    assert st == 404
    st, body, _ = call("POST", f"/documents/{did}/ask", {"question": "How many windows are shown on this floor plan?"})
    assert st == 200 and json.loads(body)["answer"]["value"] == 3, body
    st, body, _ = call("POST", "/documents", {"name": "x.dwg", "dataBase64": "AAAA"})
    assert st == 400 and "Unsupported format" in json.loads(body)["error"]
    st, body, _ = call("POST", "/documents/nope/ask", {"question": "?"})
    assert st == 404
    srv.shutdown()

print("service: all checks passed")
