"""HTTP service for the web frontend (Make_Blueprint_Talk): upload a drawing / IFC, ask questions, get explanation/1.

    pip install -r requirements.txt   (+ torch CPU, see PROJECT_NOTES.md);  export GEMINI_API_KEY=...
    python serve.py                                  # http://127.0.0.1:8765
    python serve.py --vision --min-conf-symbols 0.65 --port 8765
    python serve.py --no-llm                         # rules only (plans), IFC questions then return 503

Endpoints (JSON; the Node server forwards /api/planqa/* here):
    GET  /health                         -> {ok, agent, vision, thresholds, documents}
    POST /documents                      {name, dataBase64, page?} -> document info (status queued|processing)
    GET  /documents/<id>                 -> status, stage, error; when ready: explanation (elements, hotspots,
                                            blind spots, assets) for plans, summary for IFC
    POST /documents/<id>/ask             {question} -> {answer (explanation/1 Answer), elements, viewer_ids, ...}
    GET  /documents/<id>/answers         -> all answers of this document
    GET  /documents/<id>/assets/<file>   -> uncertainty_<channel>.png, layout.png

Standard library only (no web framework). Models are loaded on the first plan; one plan is processed at a time.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from planqa.service_core import Store  # noqa: E402

MAX_BODY = 400 * 1024 * 1024
ASSET_RE = re.compile(r"^(uncertainty_[a-z]+|layout)\.png$")


def make_handler(store: Store, meta: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "planqa/1"

        def log_message(self, fmt, *args):   # quieter than the default (one line per request)
            if not self.path.startswith("/documents/") or "/assets/" not in self.path:
                sys.stderr.write(f"[http] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

        # ---------------------------------------------------------------- helpers
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("File too large")
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def _parts(self) -> list[str]:
            return [p for p in self.path.split("?")[0].split("/") if p]

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        # ---------------------------------------------------------------- routes
        def do_GET(self):
            p = self._parts()
            try:
                if p == ["health"]:
                    return self._json(200, {**meta, "ok": True, "documents": len(store.docs)})
                if len(p) == 2 and p[0] == "documents":
                    return self._json(200, store.get(p[1]).info())
                if len(p) == 3 and p[0] == "documents" and p[2] == "answers":
                    return self._json(200, {"answers": store.get(p[1]).answers})
                if len(p) == 4 and p[0] == "documents" and p[2] == "assets" and ASSET_RE.match(p[3]):
                    f = store.get(p[1]).out_dir / p[3]
                    if not f.is_file():
                        return self._json(404, {"error": "asset not found"})
                    data = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                return self._json(404, {"error": f"unknown route {self.path}"})
            except KeyError:
                return self._json(404, {"error": "Unknown document (service restarted? send the file again)"})

        def do_POST(self):
            p = self._parts()
            try:
                body = self._body()
                if p == ["documents"]:
                    name, b64 = body.get("name"), body.get("dataBase64")
                    if not name or not b64:
                        return self._json(400, {"error": "name and dataBase64 are required"})
                    data = base64.b64decode(b64.split(",", 1)[-1])
                    doc = store.submit(name, data, int(body.get("page") or 0))
                    return self._json(202, doc.info())
                if len(p) == 3 and p[0] == "documents" and p[2] == "ask":
                    return self._json(200, store.ask(p[1], str(body.get("question") or "")))
                return self._json(404, {"error": f"unknown route {self.path}"})
            except KeyError:
                return self._json(404, {"error": "Unknown document (service restarted? send the file again)"})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except RuntimeError as e:
                return self._json(503, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="out/service", help="working folder for uploaded documents and their outputs")
    ap.add_argument("--ocr-model", default="small", choices=["tiny", "small", "medium", "auto"])
    ap.add_argument("--ocr-scale", type=int, default=1600)
    ap.add_argument("--no-tta", action="store_true", help="layout without 4x rotation averaging (4x faster)")
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--min-conf-text", type=float, default=0.6)
    ap.add_argument("--min-conf-symbols", type=float, default=0.6)
    ap.add_argument("--min-conf-rooms", type=float, default=0.8)
    ap.add_argument("--weights", default="models/model_best_val_loss_var.pkl")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--thinking", default=None, choices=["off", "low", "medium", "high", "default"])
    ap.add_argument("--vision", action="store_true", help="agent may look at crops of the drawing (view tool)")
    ap.add_argument("--free-overrides", action="store_true")
    args = ap.parse_args()

    from answer import make_agent   # same agent setup as the benchmark
    min_conf = args.min_conf if args.min_conf is not None else \
        {"texts": args.min_conf_text, "objects": args.min_conf_symbols, "rooms": args.min_conf_rooms}
    cfg = dict(tta=not args.no_tta, weights=args.weights, ocr_model=args.ocr_model, ocr_scale=args.ocr_scale,
               min_conf=min_conf)
    agent = None if args.no_llm else make_agent(args.model, args.max_steps, args.thinking, 1, args.vision,
                                                not args.free_overrides)

    def pipeline_factory():
        from planqa.pipeline import Pipeline
        return Pipeline(**cfg)

    models = {"ocr": f"PP-OCRv6_{args.ocr_model}", "layout": "cubicasa5k"}
    if agent is not None:
        models["agent"] = agent.backend.model
    store = Store(Path(args.out), agent=agent, pipeline_factory=pipeline_factory, min_conf=min_conf, models=models)
    meta = {"agent": models.get("agent"), "vision": args.vision, "thresholds": min_conf,
            "formats": ["png", "jpg", "jpeg", "webp", "pdf", "ifc"]}
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store, meta))
    print(f"planqa service on http://{args.host}:{args.port}  (agent: {models.get('agent') or 'none, rules only'})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
