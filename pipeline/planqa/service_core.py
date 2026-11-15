"""Documents and answers for the web frontend (serve.py): the same pipeline as answer.py / ask.py, kept in memory.

    store = Store(out_root, agent=make_agent(...))
    doc = store.submit("plan.png", data, page=0)      # processed in a background thread
    store.ask(doc.id, "How many doors are there?")    # -> {answer, elements, viewer_ids, ...}

Plans (PNG/JPG/PDF page): OCR + layout model -> plan_graph -> explanation/1 without answers (elements incl.
candidates below the thresholds, uncertainty maps, hotspots, blind spots). Each question: rules (qa.py) + LLM agent
on building_graph, then explain.build_answer -> one Answer of explanation/1.
IFC: ifc_api.read_ifc -> ifc_extract.build_graph -> agent; the Answer is built from the agent result (evidence ids
= table rows with guid). viewer_ids maps element ids to '#<STEP id>' (how the 3D viewer of the frontend names meshes).

Nothing is cached on disk between runs (decision: detections are produced fresh); within one server process a
document with the same bytes + page is processed once.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PLAN_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
IFC_EXT = {".ifc"}
COUNT_INTENTS = {"doors", "windows", "toilets", "sinks", "showers_bathtubs", "saunas", "bathrooms", "wc_rooms",
                 "kitchens", "bedrooms", "balconies", "floors"}
IFC_KIND = {"spaces": "space", "walls": "wall", "curtain_walls": "curtain_wall", "doors": "door", "windows": "window",
            "slabs": "slab", "roofs": "roof", "stairs": "stair", "flights": "stair_flight", "ramps": "ramp",
            "columns": "column", "beams": "beam", "railings": "railing", "footings": "footing",
            "coverings": "covering", "sanitary": "sanitary", "furniture": "furniture", "members": "member",
            "plates": "plate"}


def kind_of(name: str) -> Optional[str]:
    ext = Path(name).suffix.lower()
    return "plan" if ext in PLAN_EXT else "ifc" if ext in IFC_EXT else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:120] or "document"


@dataclass
class Doc:
    id: str
    name: str
    kind: str                      # plan | ifc
    page: int
    path: Path
    out_dir: Path
    status: str = "queued"         # queued | processing | ready | error
    error: Optional[str] = None
    stage: str = ""
    created: float = field(default_factory=time.time)
    seconds: Optional[float] = None
    # plan
    graph: Any = None
    image: Any = None
    base: Any = None               # ExplanationDoc without answers
    th: dict = field(default_factory=dict)
    bg_tables: dict = field(default_factory=dict)
    # ifc
    data: Optional[dict] = None
    index: dict = field(default_factory=dict)       # element id -> {kind, label, guid, storey}
    viewer_ids: dict = field(default_factory=dict)  # element id -> '#<STEP id>'
    answers: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def info(self) -> dict:
        d = {"id": self.id, "name": self.name, "kind": self.kind, "page": self.page, "status": self.status,
             "stage": self.stage, "error": self.error, "seconds": self.seconds, "n_answers": len(self.answers)}
        if self.status == "ready" and self.kind == "plan" and self.base is not None:
            b = json.loads(self.base.model_dump_json(exclude_none=True))
            b.pop("answers", None)
            d["explanation"] = b
            d["assets"] = {c: f"/documents/{self.id}/assets/{f}" for c, f in
                           (b.get("assets", {}).get("uncertainty_images") or {}).items()}
            if (self.out_dir / "layout.png").is_file():
                d["assets"]["layout"] = f"/documents/{self.id}/assets/layout.png"
        if self.status == "ready" and self.kind == "ifc" and self.data is not None:
            d["summary"] = {"storeys": len(self.data.get("storeys") or []),
                            "spaces": len((self.data.get("spaces") or {}).get("rows") or []),
                            "elements": {k: len(v.get("rows") or []) for k, v in (self.data.get("elements") or {}).items()},
                            "notes": (self.data.get("notes") or [])[:20]}
        return d


class Store:
    """pipeline_factory: () -> planqa.pipeline.Pipeline (loaded lazily, one plan at a time: it keeps last_* state).
    agent: planqa.agent.Agent or None (rules only for plans, IFC questions need the agent)."""

    def __init__(self, out_root: Path, agent=None, pipeline_factory=None, min_conf=None, models: dict | None = None):
        self.out_root = Path(out_root)
        self.agent, self.pipeline_factory, self.min_conf = agent, pipeline_factory, min_conf
        self.models = models or {}
        self.docs: dict[str, Doc] = {}
        self._pipe = None
        self._pipe_lock = threading.Lock()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ documents
    def submit(self, name: str, data: bytes, page: int = 0, background: bool = True) -> Doc:
        kind = kind_of(name)
        if kind is None:
            raise ValueError(f"Unsupported format: {Path(name).suffix or name} (plans: PNG/JPG/PDF, models: IFC)")
        did = hashlib.sha256(data + f"|{page}".encode()).hexdigest()[:16]
        with self._lock:
            doc = self.docs.get(did)
            if doc and doc.status != "error":
                return doc
            out_dir = self.out_root / did
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / _safe_name(name)
            path.write_bytes(data)
            doc = Doc(id=did, name=name, kind=kind, page=page, path=path, out_dir=out_dir)
            self.docs[did] = doc
        if background:
            threading.Thread(target=self._process, args=(doc,), daemon=True).start()
        else:
            self._process(doc)
        return doc

    def get(self, did: str) -> Doc:
        doc = self.docs.get(did)
        if doc is None:
            raise KeyError(did)
        return doc

    def _process(self, doc: Doc) -> None:
        t0 = time.perf_counter()
        doc.status = "processing"
        try:
            if doc.kind == "plan":
                self._process_plan(doc)
            else:
                self._process_ifc(doc)
            doc.status = "ready"
        except Exception as e:
            doc.status, doc.error = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        doc.stage = ""
        doc.seconds = round(time.perf_counter() - t0, 1)
        print(f"[planqa] {doc.name}: {doc.status} in {doc.seconds}s" + (f" ({doc.error})" if doc.error else ""),
              flush=True)

    def _process_plan(self, doc: Doc) -> None:
        from .build_graph import thresholds as norm_th
        from .explain import build_explanation
        from .plan_to_graph import to_building_graph

        with self._pipe_lock:
            if self._pipe is None:
                doc.stage = "loading models (OCR, layout)"
                if self.pipeline_factory is None:
                    raise RuntimeError("no pipeline configured")
                self._pipe = self.pipeline_factory()
            pipe = self._pipe
            doc.stage = "OCR + layout model"
            graph, _ = pipe.process(doc.path, doc.out_dir, page=doc.page, report=False)
            image, det = pipe.last_image, pipe.last_detections
            probs = getattr(pipe, "last_probs", None)
            layout_img = getattr(pipe, "last_layout_img", None)
        doc.stage = "preparing the explanation"
        base, heat = build_explanation(image, graph, det, probs, [], self.min_conf, dict(self.models))
        for c, im in heat.items():
            im.save(doc.out_dir / f"uncertainty_{c}.png")
        if layout_img is not None:
            layout_img.save(doc.out_dir / "layout.png")
        (doc.out_dir / "explanation_base.json").write_text(base.model_dump_json(indent=1, exclude_none=True),
                                                           encoding="utf-8")
        bg = json.loads(to_building_graph(graph).model_dump_json(exclude_none=True))
        tables = {f"elements.{k}": [r[0] for r in v["rows"]] for k, v in (bg.get("elements") or {}).items()}
        tables["spaces"] = [r[0] for r in (bg.get("spaces") or {}).get("rows", [])]
        doc.graph, doc.image, doc.base, doc.th, doc.bg_tables = graph, image, base, norm_th(self.min_conf), tables

    def _process_ifc(self, doc: Doc) -> None:
        from .ifc_api import read_ifc
        doc.stage = "reading IFC (ifcopenshell)"
        raw = read_ifc(doc.path, geometry=True)
        self.load_ifc_raw(doc, raw)

    def load_ifc_raw(self, doc: Doc, raw: dict) -> None:
        """raw IFC dicts -> graph, element index, viewer ids (separate for tests with synthetic models)."""
        from .ifc_extract import build_graph
        doc.stage = "building the building graph"
        graph = build_graph(raw)
        text = graph.model_dump_json(exclude_none=True)
        (doc.out_dir / "ifc_graph.json").write_text(text, encoding="utf-8")
        doc.data = json.loads(text)
        guid2eid = {e.get("guid"): e.get("eid") for e in raw.get("elements") or [] if e.get("guid")}
        tables = {"spaces": doc.data.get("spaces") or {}}
        tables.update({f"elements.{k}": v for k, v in (doc.data.get("elements") or {}).items()})
        for tname, t in tables.items():
            cols = t.get("columns") or []
            group = tname.split(".")[-1]
            for row in t.get("rows") or []:
                r = dict(zip(cols, row))
                rid = r.get("id")
                if not rid:
                    continue
                label = " · ".join(str(x) for x in (r.get("name"), r.get("number")) if x) or rid
                doc.index[rid] = {"kind": IFC_KIND.get(group, group.rstrip("s")), "label": label,
                                  "guid": r.get("guid"), "storey": r.get("storey"), "table": tname}
                eid = guid2eid.get(r.get("guid"))
                if eid is not None:
                    doc.viewer_ids[rid] = f"#{eid}"

    # ------------------------------------------------------------------ questions
    def ask(self, did: str, question: str) -> dict:
        doc = self.get(did)
        if doc.status != "ready":
            raise RuntimeError(f"Document is not ready yet ({doc.status}{': ' + doc.error if doc.error else ''})")
        question = question.strip()
        if not question:
            raise ValueError("Empty question")
        t0 = time.perf_counter()
        ans, elements = (self._ask_plan if doc.kind == "plan" else self._ask_ifc)(doc, question)
        d = json.loads(ans.model_dump_json(exclude_none=True))
        with doc.lock:
            d["id"] = f"q{len(doc.answers) + 1}"
            doc.answers.append(d)
        return {"document": doc.id, "kind": doc.kind, "answer": d, "elements": elements,
                "viewer_ids": {k: v for k, v in doc.viewer_ids.items() if k in elements},
                "image_size": doc.base.source.image_size if doc.kind == "plan" and doc.base else None,
                "agent": getattr(getattr(self.agent, "backend", None), "model", None),
                "seconds": round(time.perf_counter() - t0, 1)}

    def _ask_plan(self, doc: Doc, question: str):
        from answer import answer_items, llm_answers   # same code path as the benchmark
        from .explain import build_answer

        recs = answer_items(doc.graph, [{"question": question}])
        rec = recs[0]
        if rec.get("intent") in COUNT_INTENTS:
            rec["unit"] = "count"
        if self.agent is not None:
            llm_answers(recs, doc.graph, self.agent, doc.image if getattr(self.agent, "vision", False) else None)
        els = doc.base.elements
        ans = build_answer(len(doc.answers), rec, els, doc.bg_tables, doc.th, doc.base.blind_spots)
        if ans.value is None and not rec.get("llm_answer"):
            ans.abstained = True
            ans.abstain_reason = ("The rules do not know this question type and no LLM agent is available "
                                  "(GEMINI_API_KEY missing?)." if self.agent is None else
                                  rec.get("llm_error") or rec.get("llm_reason") or "No answer from the data.")
            ans.text = "No reliable answer possible."
            ans.summary = ans.abstain_reason
            ans.confidence, ans.confidence_label = 0.0, "niedrig"
        if rec.get("abstained") and rec.get("llm_reason"):
            ans.abstain_reason = rec["llm_reason"]
        if rec.get("llm_provenance"):
            ans.limitations.append(f"Provenance according to the agent: {rec['llm_provenance']}")
        ans.limitations += [f"Assumption: {a}" for a in rec.get("llm_assumptions") or []]
        ids = self._referenced(ans)
        elements = {i: json.loads(els[i].model_dump_json(exclude_none=True)) for i in ids if i in els}
        return ans, elements

    def _ask_ifc(self, doc: Doc, question: str):
        from .explain import _first_sentence, _ids_of_step, _result_short, _tool_title
        from .explanation_schema import (Answer, Element, Evidence, Geometry, SourceAnswer, Step)

        if self.agent is None:
            raise RuntimeError("IFC questions need the LLM agent: set GEMINI_API_KEY and install google-genai.")
        r = self.agent.ask(doc.data, question)
        known = set(doc.index)
        ev_ids = [x for x in dict.fromkeys(str(e) for e in r.evidence or []) if x in known]
        refs = [str(x) for x in r.evidence or [] if str(x) not in known]
        counting = r.value is not None and (r.unit in (None, "", "count", "pcs", "Stk"))
        evidence = [Evidence(element=x, role="counted" if counting else "context", order=n + 1 if counting else None,
                             reason=doc.index[x]["label"]) for n, x in enumerate(ev_ids)]
        steps, first_seen = [], {}
        for t in r.trace or []:
            n = len(steps) + 1
            ids = _ids_of_step(t, known)
            if t["tool"] == "final_answer":
                steps.append(Step(n=n, kind="final", tool="final_answer", title=f"Antwort: {r.answer or r.value}",
                                  result=_first_sentence((t.get("args") or {}).get("reason"), 200) or None,
                                  ids=ids or ev_ids))
                continue
            steps.append(Step(n=n, kind="read" if t["tool"] == "overview" else "tool", tool=t["tool"],
                              args=t.get("args") or None, title=_tool_title(t), result=_result_short(t), ids=ids))
            for x in ids:
                first_seen.setdefault(x, n)
        conf = float(r.confidence or 0)
        limitations = []
        if r.provenance:
            limitations.append(f"Provenance according to the agent: {r.provenance}")
        limitations += [f"Assumption: {a}" for a in r.assumptions or []]
        if r.error:
            limitations.append(f"Agent error: {r.error}")
        ans = Answer(id="q0", question=question, value=r.value, unit=r.unit,
                     text=r.answer or ("No reliable answer possible." if r.abstained else str(r.value)),
                     summary=_first_sentence(r.reason) or (r.answer or ""), confidence=round(conf, 2),
                     confidence_label="hoch" if conf >= 0.8 else "mittel" if conf >= 0.5 else "niedrig",
                     abstained=bool(r.abstained or (r.value is None and not r.answer)),
                     abstain_reason=(r.reason or r.error) if (r.abstained or r.error) else None,
                     agreement="single",
                     sources={"agent": SourceAnswer(value=r.value, text=r.answer, explanation=r.reason, model=r.model)},
                     evidence=evidence, references=list(dict.fromkeys(refs)), steps=steps, limitations=limitations)
        ids = self._referenced(ans)
        elements = {}
        for i in ids:
            if i in doc.index:
                x = doc.index[i]
                elements[i] = json.loads(Element(id=i, kind=x["kind"], label=x["label"], status="kept",
                                                 geometry=Geometry(guid=x["guid"], storey=x["storey"]))
                                         .model_dump_json(exclude_none=True))
        return ans, elements

    @staticmethod
    def _referenced(ans) -> list[str]:
        ids = [e.element for e in ans.evidence]
        for s in ans.steps:
            ids += s.ids
        return list(dict.fromkeys(ids))
