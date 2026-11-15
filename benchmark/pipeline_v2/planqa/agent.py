"""LLM agent that answers questions from a generated JSON (ifc_graph.json or plan_graph.json).

The model never sees the whole JSON. It gets an overview and queries the data with tools
(tables with filters, aggregates, text search, a calculator) and must finish with
final_answer(...) citing the ids it used. Every tool call is kept in the trace, so an
answer can be checked step by step (explainability), and "not in the model" is an
explicit, reasoned outcome (abstention) instead of a guess.

Backends: GeminiBackend (google-genai, needs GEMINI_API_KEY) and ScriptedBackend (tests).
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

MAX_RESULT_CHARS = 4000
MAX_VIEWS = 3   # images stay in the history and are re-sent every step: keep it small
HEAVY_COLUMNS = {"box_px", "polygon_px", "type_reason", "centroid_xy", "guid", "label_ids", "area_ids"}   # only when asked for


# ======================================================================== tools on the JSON

class JsonTools:
    """Read-only query tools over one JSON document. Tables are {"columns": [...], "rows": [[...]]}
    (ifc_graph) or lists of dicts (plan_graph); both are addressed by a dotted path."""

    def __init__(self, data: dict, image=None):
        from .plan_to_graph import as_building_graph
        self.image, self.n_views = image, 0
        self.data = data = as_building_graph(data)   # plan_graph.json -> building-graph
        sv = str(data.get("schema_version", ""))
        self.kind = "ifc" if sv.startswith("ifc-graph") or (sv.startswith("building-graph") and
                                                            (data.get("source") or {}).get("kind") == "ifc") \
            else "plan" if sv.startswith("building-graph") else "json"
        self.tables = self._find_tables(data, "")

    # ---------- helpers
    def _find_tables(self, x, path) -> dict[str, list[str]]:
        out = {}
        if isinstance(x, dict):
            if isinstance(x.get("columns"), list) and isinstance(x.get("rows"), list):
                return {path: x["columns"]}
            for k, v in x.items():
                out.update(self._find_tables(v, f"{path}.{k}" if path else k))
        elif isinstance(x, list) and x and all(isinstance(i, dict) for i in x[:5]) and len(x) >= 2:
            cols = list(dict.fromkeys(k for i in x[:50] for k in i))
            if cols:
                out[path] = cols
        return out

    def _resolve(self, path: str):
        x = self.data
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", path or ""):
            if part.startswith("["):
                x = x[int(part[1:-1])]
            elif isinstance(x, dict) and part in x:
                x = x[part]
            else:
                raise KeyError(f"path '{path}' not found at '{part}'")
        return x

    def _rows(self, table: str) -> tuple[list[str], list[dict]]:
        t = self._resolve(table)
        if isinstance(t, dict) and "columns" in t:
            return t["columns"], [dict(zip(t["columns"], r)) for r in t["rows"]]
        if isinstance(t, list):
            return self.tables.get(table) or list(dict.fromkeys(k for i in t for k in i)), [dict(i) for i in t]
        raise KeyError(f"'{table}' is not a table; tables: {', '.join(self.tables)}")

    @staticmethod
    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and re.fullmatch(r"-?\d+(?:[.,]\d+)?", v.strip()):
            return float(v.strip().replace(",", "."))
        return None

    def _match(self, row: dict, cond: dict) -> bool:
        col, op, val = cond.get("column"), (cond.get("op") or "=").lower(), cond.get("value")
        v = row.get(col)
        if op in ("is_null", "null"):
            return v is None
        if op in ("not_null", "is_not_null"):
            return v is not None
        if op == "contains":
            needle = _norm(str(val))
            if isinstance(v, (list, tuple)):
                return any(needle == _norm(str(i)) or needle in _norm(str(i)) for i in v)
            if isinstance(v, dict):
                return any(needle in _norm(str(k)) for k in v)
            return v is not None and needle in _norm(str(v))
        if op == "in":
            vals = val if isinstance(val, list) else [s.strip() for s in str(val).split(",")]
            return _norm(str(v)) in {_norm(str(x)) for x in vals}
        a, b = self._num(v), self._num(val)
        if op in ("=", "==", "!="):
            if isinstance(v, bool) or str(val).lower() in ("true", "false"):
                eq = str(v).lower() == str(val).lower()
            elif a is not None and b is not None:
                eq = abs(a - b) < 1e-9
            else:
                eq = _norm(str(v)) == _norm(str(val))
            return eq if op != "!=" else not eq
        if a is None or b is None:
            return False
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}.get(op, False)

    @staticmethod
    def _where(where) -> list[dict]:
        """Conditions as dicts. Models sometimes send them as JSON strings or 'col op value' text."""
        if not where:
            return []
        if isinstance(where, (str, dict)):
            where = [where]
        out = []
        for c in where:
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except ValueError:
                    m = re.match(r"^\s*([\w.]+)\s*(==|!=|<=|>=|=|<|>|contains|in|is_null|not_null)\s*(.*?)\s*$", c, re.I)
                    if not m:
                        raise ValueError(f"condition '{c}' not understood; use {{column, op, value}}")
                    c = {"column": m[1], "op": m[2].lower(), "value": m[3].strip("'\"")}
            if not isinstance(c, dict):
                try:
                    c = dict(c)
                except (TypeError, ValueError):
                    raise ValueError(f"condition {c!r} not understood; use {{column, op, value}}")
            out.append(c)
        return out

    def _filter(self, table, where):
        cols, rows = self._rows(table)
        where = self._where(where)
        bad = [c.get("column") for c in where if c.get("column") not in cols]
        if bad:
            raise KeyError(f"unknown column(s) {bad} in '{table}'; columns: {cols}")
        return cols, [r for r in rows if all(self._match(r, c) for c in where)]

    # ---------- tools (names and signatures are what the model sees)
    def overview(self) -> dict:
        d = self.data
        out: dict[str, Any] = {"kind": self.kind,
                               "tables": {p: {"columns": c, "rows": len(self._rows(p)[1])} for p, c in self.tables.items()}}
        if self.kind in ("ifc", "plan"):
            f = d.get("facts") or {}
            areas = {k: v for k, v in (f.get("areas") or {}).items() if k not in ("by_name", "by_category", "by_unit")}
            out.update(source=d.get("source"), building=d.get("building"),
                       facts_counts=f.get("counts"), facts_areas=areas, facts_how=f.get("how"),
                       facts_keys=list((f.get("areas") or {}).keys()),
                       storeys=[{k: s.get(k) for k in ("id", "name", "elevation", "height", "n_spaces", "net_floor_area",
                                                        "gross_floor_area", "counts")} for s in d.get("storeys", [])],
                       summary_keys=list(d.get("summary", {})), summary_counts=d.get("summary", {}).get("counts"),
                       n_types=len(d.get("types", [])), inventory_groups=list(d.get("inventory", {})),
                       mep_keys=list(d.get("mep", {})), n_relations=len(d.get("relations", [])),
                       relation_kinds=sorted({r.get("rel") for r in d.get("relations", [])}),
                       schedules=list(d.get("schedules", {})), legends=list(d.get("legends", {})), notes=d.get("notes"))
        else:
            out["keys"] = {k: type(v).__name__ for k, v in d.items()}
        return out

    def get(self, path: str) -> Any:
        return self._resolve(path)

    def query_table(self, table: str, where: list | None = None, columns: list | None = None,
                    sort_by: str | None = None, descending: bool = False, limit: int = 25) -> dict:
        cols, rows = self._filter(table, where)
        if sort_by:
            if sort_by not in cols:
                raise KeyError(f"unknown sort column '{sort_by}'; columns: {cols}")
            rows.sort(key=lambda r: (self._num(r.get(sort_by)) is None, self._num(r.get(sort_by)) or 0,
                                     str(r.get(sort_by))), reverse=descending)
            if descending:  # keep missing values last
                rows.sort(key=lambda r: r.get(sort_by) is None)
        show = [c for c in columns if c in cols] if columns else [c for c in cols if c not in HEAVY_COLUMNS]
        limit = max(1, min(int(limit or 25), 200))
        return {"n_matched": len(rows), "columns": show, "rows": [[r.get(c) for c in show] for r in rows[:limit]],
                "truncated": len(rows) > limit}

    def aggregate(self, table: str, op: str, column: str | None = None, where: list | None = None,
                  group_by: str | None = None) -> dict:
        cols, rows = self._filter(table, where)
        op = op.lower()
        if op not in ("count", "sum", "avg", "min", "max", "count_distinct"):
            raise ValueError("op must be count, sum, avg, min, max or count_distinct")
        if op != "count" and column not in cols:
            raise KeyError(f"column '{column}' not in {cols}")
        if group_by and group_by not in cols:
            raise KeyError(f"group_by '{group_by}' not in {cols}")

        def agg(rs):
            if op == "count":
                return len(rs)
            vals = [r.get(column) for r in rs if r.get(column) is not None]
            if op == "count_distinct":
                return len({json.dumps(v, sort_keys=True) for v in vals})
            nums = [self._num(v) for v in vals if self._num(v) is not None]
            if not nums:
                return None
            res = {"sum": sum(nums), "avg": sum(nums) / len(nums), "min": min(nums), "max": max(nums)}[op]
            return round(res, 4)

        res = {"op": op, "column": column, "n_rows": len(rows),
               "n_missing_values": sum(r.get(column) is None for r in rows) if column else 0}
        if group_by:
            groups: dict[str, list] = {}
            for r in rows:
                groups.setdefault(str(r.get(group_by)), []).append(r)
            res["groups"] = {k: agg(v) for k, v in sorted(groups.items())}
        else:
            res["value"] = agg(rows)
        return res

    def search(self, text: str, limit: int = 20) -> dict:
        """Case/accent-insensitive search in all strings; hits inside tables report table + row id."""
        needle, hits = _norm(text), []
        for path, cols in self.tables.items():
            for r in self._rows(path)[1]:
                for c, v in r.items():
                    if isinstance(v, str) and needle in _norm(v):
                        hits.append({"table": path, "row_id": r.get("id"), "column": c, "value": v})
        tabled = set(self.tables)

        def walk(x, p):
            if p in tabled:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{p}.{k}" if p else k)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{p}[{i}]")
            elif isinstance(x, str) and needle in _norm(x):
                hits.append({"path": p, "value": x})
        walk(self.data, "")
        return {"n_hits": len(hits), "hits": hits[:max(1, min(int(limit or 20), 100))]}

    def view(self, id: str | None = None, box_px: list | None = None, zoom: float = 1.0) -> dict:
        """Crop of the original drawing around a room / text / symbol id, a pixel box, or 'plan' (whole sheet)."""
        if self.image is None:
            raise ValueError("no drawing available")
        if self.n_views >= MAX_VIEWS:
            raise ValueError(f"view limit ({MAX_VIEWS}) reached: answer from the data and the views so far")
        W, H = self.image.size
        if id in (None, "", "plan") and not box_px:
            box, what = [0, 0, W, H], "whole drawing"
        elif box_px:
            box, what = [int(float(v)) for v in list(box_px)[:4]], "box"
        else:
            box = self._box_of(str(id))
            what = str(id)
        x0, y0, x1, y1 = box
        m = max(24, int(0.25 * max(x1 - x0, y1 - y0) / max(float(zoom or 1), 0.25)))   # context around it
        x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"box {box} is empty")
        crop = self.image.crop((x0, y0, x1, y1)).convert("RGB")
        long = max(crop.size)
        if what == "whole drawing":
            f, resolution = min(1.0, 1024 / long), "overview"
            note = ("OVERVIEW at reduced resolution: use it for the layout (where things are, how many separate "
                    "drawings / storeys), NOT for counting symbols or reading small text; request a crop for details")
        else:
            f = min(3.0, max(1.0, 384 / long))   # never below the original resolution ...
            resolution = "full"
            note = "detail crop at full resolution"
            if long * f > 1024:                  # ... unless the box is very large
                f, resolution = 1024 / long, "reduced"
                note = "box too large for full resolution: request a smaller box (or zoom > 1) for details"
        crop = crop.resize((max(1, int(crop.width * f)), max(1, int(crop.height * f))))
        import io
        buf = io.BytesIO()
        crop.save(buf, "PNG")
        self.n_views += 1
        return {"shown": what, "crop_px": [x0, y0, x1, y1], "scale": round(f, 2), "resolution": resolution,
                "views_left": MAX_VIEWS - self.n_views,
                "note": note + "; image follows; drawing coordinates = crop_px origin + image px / scale",
                "_image": buf.getvalue()}

    def ids(self) -> set[str]:
        out = {str(s.get("id")) for s in self.data.get("storeys", [])}
        for path in self.tables:
            cols, rows = self._rows(path)
            if "id" in cols:
                out |= {str(r.get("id")) for r in rows if r.get("id") is not None}
        return out

    def _box_of(self, id_: str) -> list[int]:
        for path in self.tables:
            cols, rows = self._rows(path)
            if "box_px" in cols and "id" in cols:
                for r in rows:
                    if str(r.get("id")) == id_ and r.get("box_px"):
                        return list(r["box_px"])
        for s in self.data.get("storeys", []):
            if s.get("id") == id_ and s.get("box_px"):
                return list(s["box_px"])
        raise KeyError(f"no box for id '{id_}' (use a room r*, text t*, symbol o*, storey L* id, a box_px or 'plan')")

    def calculate(self, expression: str) -> dict:
        return {"expression": expression, "result": _safe_eval(expression)}


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode() if s else ""
    return re.sub(r"[\s\"'„“”‚‘’]+", " ", s).strip().lower()


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": lambda *a: sum(a[0] if len(a) == 1 else a),
          "sqrt": math.sqrt}


def _safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in calculate: {ast.dump(n)[:60]}")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval"))


# ======================================================================== tool declarations

COND = {"type": "object", "properties": {
    "column": {"type": "string"},
    "op": {"type": "string", "enum": ["=", "!=", "<", "<=", ">", ">=", "contains", "in", "is_null", "not_null"]},
    "value": {"type": "string", "description": "compared numerically when both sides are numbers; "
                                               "booleans as true/false; 'in' takes a comma-separated list"}},
    "required": ["column", "op"]}

DECLARATIONS = [
    {"name": "overview", "description": "Structure of the document: tables with columns and row counts, storeys, "
                                        "precomputed summary keys, notes about missing data. Call this first.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get", "description": "Return the JSON value at a dotted path, e.g. 'facts.areas.by_category', "
                                   "'facts.areas.by_name', 'summary.walls', 'storeys[0]', 'building', 'types', "
                                   "'mep.systems', 'legends', 'notes'.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "query_table", "description": "Rows of a table (path from overview, e.g. 'spaces', 'elements.doors', "
                                           "'elements.sanitary', 'annotations') filtered by conditions (AND), optionally "
                                           "sorted. Use op 'contains' for list columns such as categories.",
     "parameters": {"type": "object", "properties": {
         "table": {"type": "string"}, "where": {"type": "array", "items": COND},
         "columns": {"type": "array", "items": {"type": "string"}, "description": "columns to return (default all)"},
         "sort_by": {"type": "string"}, "descending": {"type": "boolean"},
         "limit": {"type": "integer", "description": "max rows returned (default 30, max 200)"}},
         "required": ["table"]}},
    {"name": "aggregate", "description": "count / sum / avg / min / max / count_distinct of a column over the "
                                         "(filtered) rows of a table, optionally grouped by another column.",
     "parameters": {"type": "object", "properties": {
         "table": {"type": "string"},
         "op": {"type": "string", "enum": ["count", "sum", "avg", "min", "max", "count_distinct"]},
         "column": {"type": "string"}, "where": {"type": "array", "items": COND}, "group_by": {"type": "string"}},
         "required": ["table", "op"]}},
    {"name": "search", "description": "Find a name or text anywhere (case/accent-insensitive substring), e.g. a "
                                      "room name, storey name, material or type. Returns table + row id or JSON path.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["text"]}},
    {"name": "calculate", "description": "Evaluate arithmetic exactly (+ - * / ** round abs min max sum sqrt). "
                                         "Use it for every computation instead of mental arithmetic.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "final_answer", "description": "Finish. Exactly one call, after the data was checked with tools.",
     "parameters": {"type": "object", "properties": {
         "answer": {"type": "string", "description": "complete answer in the language of the question"},
         "value": {"type": "number", "description": "the single number asked for (count, m², m, m³), if any"},
         "unit": {"type": "string", "description": "count | m2 | m | m3 | ..."},
         "evidence": {"type": "array", "items": {"type": "string"},
                      "description": "ids / paths the answer is based on (e.g. S12, W5, L0, T3, summary.walls)"},
         "confidence": {"type": "number", "description": "0..1"},
         "abstained": {"type": "boolean", "description": "true if neither a value from the model nor a derived value "
                                                         "or estimate can be given"},
         "provenance": {"type": "string", "enum": ["stated", "derived", "estimate", "not_available"],
                        "description": "where the main value / statement comes from (see rules)"},
         "assumptions": {"type": "array", "items": {"type": "string"},
                         "description": "assumptions behind a derived value or an estimate"},
         "reason": {"type": "string", "description": "how the answer was derived, or why it is not answerable"},
         "changes": {"type": "array", "description": (
             "REQUIRED when `value` differs from the given estimate / facts value: one entry per item you add to or "
             "remove from the detected ones. Net (adds - removes) must equal value - estimate. Without valid changes "
             "the estimate is kept."), "items": {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["add", "remove"]},
                 "item": {"type": "string", "description": "what, e.g. 'door', 'balcony', 'sink'"},
                 "where": {"type": "string", "description": "id of the data row (r12, o7, t3) and/or the view that "
                                                            "shows it ('view:r12', 'view:box'); not 'view:plan'"},
                 "reason": {"type": "string"}}, "required": ["action", "item", "where"]}}},
         "required": ["answer", "confidence", "reason"]}},
]

VIEW_DECL = {"name": "view", "description": (
    "Look at the original drawing: a crop around a room (r*), text (t*), symbol (o*) or storey (L*) id, around a "
    "pixel box_px, or the whole sheet (id 'plan'). Use it to read a label the OCR missed or misspelt, to check one "
    f"doubtful room or symbol, or to see how many storeys are drawn. At most {MAX_VIEWS} views per question."),
    "parameters": {"type": "object", "properties": {
        "id": {"type": "string", "description": "r*, t*, o*, L* id, or 'plan' for the whole drawing"},
        "box_px": {"type": "array", "items": {"type": "number"}, "description": "[x0, y0, x1, y1] instead of id"},
        "zoom": {"type": "number", "description": ">1 = less context around the box (default 1)"}}}}

VISION_NOTES = """
You can look at the drawing with `view`. The data is the primary source; a view is a check, not a recount:
- good for: reading a room label that is missing or garbled in the data (rooms without categories, annotations that
  look like misspelt labels), checking whether a doubtful room is e.g. a sauna/WC/balcony, a fixture in one room, and
  how many storeys / storey drawings the sheet shows (view 'plan');
- good candidates: spaces without categories or with issues no_label_text / unlabelled_outdoor_not_counted_as_balcony,
  annotations whose text looks like a garbled room label, and storeys (one detected outline but several drawn?);
- not for recounting many small symbols (windows, doors) by eye: keep the detected count unless one detail view shows a
  concrete error, and then correct only that error.
- 'plan' is a low-resolution overview for the layout only; details need a crop (id, box_px, zoom).
- A final value that differs from the estimate is only accepted with `changes`: one add/remove per item, each citing the
  data row and/or the detail view that shows it, adding up to the difference. Otherwise the estimate is kept."""

SYSTEM = """You answer questions about a building using ONLY a JSON document that you query with tools.
You never see the whole document; call overview first, then query precisely.

Rules
- Every number comes from the data or from `calculate`; never present a guess as data.
- Say where the answer comes from, in the answer text itself, and set `provenance` for the main value:
  stated = written in the model (quantities, properties, names, element counts);
  derived = computed or interpreted from the data: values the notes / *_src columns mark as derived or estimated
    (geometry, inferred ext, estimated gross areas), a construction read from a type name, sums you computed;
  estimate = the model lacks it, but a reasonable value follows from model data plus explicit assumptions
    (e.g. 2 knobs per door, wall length x storey height): give the assumptions and the calculation (`calculate`);
  not_available = it cannot be answered from the model at all.
- Prefer precomputed values: `facts` (standard counts and areas, computed the same way for every source; overview
  shows them and `facts.how` says how), per-storey aggregates in `storeys`, `summary`; otherwise aggregate the tables.
- Shared vocabulary: spaces.categories: bedroom, living, dining, kitchen, bathroom, wc, sauna, hallway, staircase,
  elevator, storage, walk_in_closet, utility, technical, shaft, office, meeting, laboratory, balcony, terrace, garage;
  a combined room has several (OH+K = living + kitchen). spaces.use is the coarse group (residential, restroom,
  circulation, storage, technical, office, meeting, laboratory, outdoor, parking, other). elements.sanitary.category:
  toilet, urinal, washbasin, sink, shower, bathtub, bidet. A missing column or null = unknown, not zero.
- Rows with `conf` come from detection (0..1); rows with src=inferred are assumptions: say so in the answer.
- Rooms are often asked as 'NAME (NUMBER)': match `name` and `number` of the spaces table (use search if unsure).
- Storeys are referenced by name in questions and by id (L0, L1, ...) in the tables.
- If what is asked is not directly in the document (see `notes`, missing columns, null values), start the answer
  with that ("Not directly in the model: ..."), then give a derived value or an estimate if one is reasonable,
  clearly labelled, with its basis. abstained=true only if you can give neither. A partial answer must say which part
  is missing.
- Cite the ids / paths you used as evidence. Answer in the language of the question; numbers in SI units.
- Be efficient: the overview (with `facts`) is already in the question message, do not call overview again.
  For a standard count, `facts.counts` is the starting answer: check only what could change it (rows with low conf,
  src=inferred, issues, labels without detected room, OCR texts that look like misspelt labels), with filters and
  few columns, then answer. Usually 1-4 tool calls are enough; never page through whole tables.
- If an estimate is given with the question and your count differs from it, list every added / removed item in
  `changes` (citing the data row id or the detail view), adding up to the difference; otherwise the estimate is kept.
- Finish with exactly one final_answer call.
{kind_notes}"""

KIND_NOTES = {
    "ifc": """
Document: building-graph from an IFC model. SI units (m, m², m³). Tables have `columns` + `rows`.
Spatial tree: building -> storeys (L*) -> spaces (S*); every element row has `storey` and, if known, `space`.
`types` (T*) hold materials, layers, thickness, profile and Pset_*Common properties (fire rating, IsExternal, ...);
elements reference them in column `type`. Booleans ext / load_bearing may be null = unknown.
`relations`: adjacent (spaces sharing a wall `via`), connected_via (door `via`, b may be OUTSIDE),
faces_outside (window `via`), connects_storeys (stair / elevator `via`). Doors/windows have `host` wall.
elements.sanitary / elements.furniture list single items; `inventory` aggregates MEP / other per type.
Spaces: `area_src` says where net_area comes from (a quantity / property of the model, or 'geometry' = derived);
spaces with `contains` are gross-area / zone spaces holding other spaces: they are spaces of the model too; when they
matter (largest space, totals, lists) say whether you included them. `ext_src` on walls / doors: ext derived from the
geometry (empty = IsExternal stated in the model). Beams: `spaces` = {space: share of the beam above it}; a beam spans
a room when it crosses it (large share), a small share means it only touches the edge of the room.
Storey gross_floor_area with gross_floor_area_src 'estimate…' is NOT in the model: call it an estimate.""",
    "plan": """
Document: building-graph from a floor-plan drawing (OCR + layout model). Coordinates (box_px) are pixels.
spaces: rooms r* (src room_polygon = class of the layout model, room_label = the label written in the room wins) and
room labels t* outside every detected room (issue label_without_detected_room). net_area: area_src=label is written
in the drawing, area_src=estimate:* is pixel area x plan scale (say it is an estimate). number = room number read
in the room (1.04); unit = apartment marker (A, App. B) inside or nearest (issue unit_from_nearest_marker).
Issues on spaces say how a room was recognised: ocr_fuzzy:<word> (misspelt label matched), label_from_text:<t>
(category from another text in the room, e.g. OH+K), second_label_in_room:<r> (two rooms merged by the layout),
unlabelled_outdoor_not_counted_as_balcony,
reserved_room_VARAUS (counted as that room type), no_label_text (room type only from the layout model).
annotations: OCR texts t* with kind (room_label, area, dimension, opening_size = door/window size marks like 9x21 or
A/12x14: their number is a cross-check for doors/windows), conf, space. elements: doors, windows, sanitary,
furniture from symbols o* with conf and space; walls w* = straight runs of the layout Wall/Railing class (not one
row per wall); elements.rejected = detections below threshold, never count them; storeys: one per drawn floor outline (or per written storey name).
Detections below source.extraction.thresholds were dropped (counts in dropped): counts can be too low.
Room labels are abbreviations (mostly Finnish): bathroom KH, KPH, PH, PESUH, SH | WC | sauna S, SAUNA, LH |
kitchen K, KK, KT, TK, AVOK | bedroom MH, MH1 ... | living OH | hallway ET | storage VAR | walk-in closet VH |
utility KHH | balcony P, PARVEKE, LASITETTU PARVEKE | terrace TERASSI. An OCR error can hide a label: check
annotations near unlabelled rooms (search) before concluding. One symbol = one item; a double sink counts once.
There is no shower symbol: showers are inferred (src=inferred), say so. Counts asked for are for the whole drawing.
A rule-based estimate may be given with the question: verify it against the data and correct it if needed.""",
    "json": "",
}


# ======================================================================== agent loop

@dataclass
class AgentResult:
    question: str
    answer: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    reason: str = ""
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    model: str = ""
    seconds: float = 0.0
    tokens: dict = field(default_factory=dict)
    error: Optional[str] = None
    changes: list = field(default_factory=list)
    provenance: Optional[str] = None     # stated | derived | estimate | not_available
    assumptions: list = field(default_factory=list)
    estimate: Optional[float] = None
    override: Optional[str] = None       # accepted | rejected: <why>   (value != estimate)

    def dict(self):
        return asdict(self)


class Agent:
    def __init__(self, backend, max_steps: int = 14):
        self.backend, self.max_steps = backend, max_steps

    def ask(self, data: dict, question: str, context: str | None = None, image=None,
            estimate: float | None = None) -> AgentResult:
        """context: extra information for the model, e.g. the answer of the rule-based baseline.
        image: the drawing (PIL image) for plans -> enables the view tool (crops of the drawing).
        estimate: the value from the data (rules / facts) for a count question. A different final value is only
        accepted with `changes` that cite concrete rows or detailed views and add up to the difference."""
        tools = JsonTools(data, image=image)
        res = AgentResult(question=question, model=getattr(self.backend, "model", ""))
        t0 = time.perf_counter()
        ov = tools.overview()   # given up front: saves a round trip, the model starts informed
        user = (f"Question: {question}" + (f"\n\n{context}" if context else "") +
                "\n\nDocument overview (result of overview()):\n" + json.dumps(ov, ensure_ascii=False, default=str))
        res.trace.append({"tool": "overview", "args": {}, "result": "(in the question message)"})
        decls = DECLARATIONS + ([VIEW_DECL] if tools.image is not None else [])
        system = SYSTEM.format(kind_notes=KIND_NOTES[tools.kind] + (VISION_NOTES if tools.image is not None else ""))
        state = self.backend.start(system, decls, user)
        try:
            for step in range(self.max_steps):
                res.steps = step + 1
                if step == self.max_steps - 1:
                    state["force_final"] = True   # last round: the model may only call final_answer
                reply = self.backend.generate(state)
                for k, v in (reply.get("usage") or {}).items():
                    res.tokens[k] = res.tokens.get(k, 0) + (v or 0)
                calls = reply.get("calls") or []
                if not calls:   # model answered in plain text: accept it, but mark it
                    res.answer = (reply.get("text") or "").strip() or None
                    res.reason = "model answered without final_answer"
                    res.confidence = 0.3
                    break
                outputs, done = [], False
                for name, args in calls:
                    args = dict(args or {})
                    if name == "final_answer":
                        self._finish(res, args)
                        res.trace.append({"tool": name, "args": args})
                        outputs.append((name, {"ok": True}))
                        done = True
                        continue
                    out = _call_tool(tools, name, args)
                    res.trace.append({"tool": name, "args": args,
                                      "result": _preview({k: v for k, v in out.items() if k != "_image"}, 600)})
                    outputs.append((name, out))
                if done:
                    break
                if step == self.max_steps - 2:
                    outputs.append(("_system", {"note": "Last step: call final_answer now with what you found; "
                                                        "say what is missing (abstained=true if the core is missing)."}))
                self.backend.add_results(state, outputs)
            else:
                res.error = "step limit reached without final_answer"
        except Exception as e:  # network / quota / malformed response: keep the trace
            res.error = f"{type(e).__name__}: {e}"
        if estimate is not None:
            self._check_override(res, float(estimate), tools)
        res.seconds = round(time.perf_counter() - t0, 2)
        return res

    @staticmethod
    def _check_override(res: AgentResult, estimate: float, tools: "JsonTools") -> None:
        """A count that differs from the data must be explained item by item (explainability, and no recounting
        by eye): every change cites an existing row id or a detailed view, and the changes add up."""
        res.estimate = estimate
        if res.value is None or round(res.value) == round(estimate):
            return
        viewed = {str(t["args"].get("id") or "box") for t in res.trace if t["tool"] == "view"
                  and '"error"' not in t.get("result", "")}
        viewed.discard("plan")   # the overview is too coarse to justify a change
        ids = tools.ids()
        problems, net = [], 0
        for c in res.changes:
            act, where = str(c.get("action", "")).lower(), str(c.get("where", ""))
            refs = set(re.findall(r"view:([\w.\-]+)", where))
            plain = {w for w in re.findall(r"[\w.\-]+", re.sub(r"view:[\w.\-]+", " ", where)) if w in ids}
            if act not in ("add", "remove"):
                problems.append(f"change without add/remove: {c}")
                continue
            if not plain and not (refs & viewed):
                problems.append(f"'{c.get('item')}' cites neither a data row nor a detailed view ({where or '-'})")
            net += 1 if act == "add" else -1
        diff = round(res.value) - round(estimate)
        if not res.changes:
            problems.append("no changes listed")
        elif net != diff:
            problems.append(f"changes add up to {net:+d}, value differs by {diff:+d}")
        if problems:
            res.override = "rejected: " + "; ".join(problems)
            res.reason = (f"[Estimate {estimate:g} kept, the agent's {res.value:g} was not backed item by item: "
                          f"{'; '.join(problems)}] " + (res.reason or ""))
            res.value = estimate
        else:
            res.override = "accepted"

    @staticmethod
    def _finish(res: AgentResult, a: dict) -> None:
        res.answer = a.get("answer")
        v = a.get("value")
        res.value = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        res.unit = a.get("unit")
        res.evidence = [str(e) for e in (a.get("evidence") or [])]
        res.confidence = float(a.get("confidence") or 0)
        res.abstained = bool(a.get("abstained"))
        res.reason = a.get("reason") or ""
        res.provenance = a.get("provenance")
        res.assumptions = [str(x) for x in (a.get("assumptions") or [])]
        ch = a.get("changes") or []
        res.changes = [dict(c) if not isinstance(c, dict) else c for c in (ch if isinstance(ch, list) else [ch])
                       if isinstance(c, dict) or hasattr(c, "keys")]


def _call_tool(tools: JsonTools, name: str, args: dict) -> dict:
    fn = getattr(tools, name, None) if name in {d["name"] for d in DECLARATIONS + [VIEW_DECL]} else None
    if name == "view" and tools.image is None:
        fn = None
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        out = fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except (KeyError, ValueError, IndexError, ZeroDivisionError, SyntaxError) as e:
        return {"error": str(e).strip("'\"")}
    except Exception as e:   # any other tool failure goes back to the model, it can retry differently
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}
    img = out.pop("_image", None) if isinstance(out, dict) else None
    text = json.dumps(out, ensure_ascii=False, default=str)
    if len(text) > MAX_RESULT_CHARS:
        return {"truncated": True, "hint": "result too long: filter, select columns or lower limit",
                "partial": text[:MAX_RESULT_CHARS]}
    out = out if isinstance(out, dict) else {"result": out}
    if img is not None:
        out["_image"] = img   # PNG bytes for the backend, never shown as text
    return out


def _preview(x, n):
    s = json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


# ======================================================================== backends

class ScriptedBackend:
    """Replays a fixed list of tool calls (offline tests). script: [[(name, args), ...], ...] per step."""

    model = "scripted"

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def start(self, system, decls, user):
        return {"i": 0}

    def generate(self, state):
        step = self.script[state["i"]] if state["i"] < len(self.script) else []
        state["i"] += 1
        return {"calls": step, "usage": {"calls": 1}}

    def add_results(self, state, outputs):
        self.seen.append(outputs)


class GeminiBackend:
    """Gemini via google-genai with manual function calling (pip install google-genai; GEMINI_API_KEY)."""

    PREFERRED = "gemini-flash-latest"
    _SKIP = ("lite", "image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "native")

    def __init__(self, model: str | None = None, retries: int = 5, thinking: str | None = None):
        from google import genai
        self.thinking = (thinking or os.environ.get("GEMINI_THINKING") or "low").lower()   # off|low|medium|high|default
        # v1beta explicitly: aliases like gemini-flash-latest, preview models, function calling with
        # thought signatures only exist there (v1 -> "404 NOT_FOUND ... not found for API version v1").
        self.api_version = os.environ.get("GEMINI_API_VERSION", "v1beta")
        self.client = genai.Client(http_options={"api_version": self.api_version})  # one client per run
        self.retries = retries
        self.model = self._resolve_model(model or os.environ.get("GEMINI_MODEL"))

    def _resolve_model(self, wanted: str | None) -> str:
        """Check the model against ListModels; if it is missing, take the newest available Flash model.
        Fails here once instead of 404 on every single question."""
        try:
            names = [n.removeprefix("models/") for n in self.list_models()]
        except Exception as e:  # listing not allowed / offline: keep what was asked for
            print(f"LLM: ListModels failed ({type(e).__name__}: {e}), using {wanted or self.PREFERRED} unchecked")
            return wanted or self.PREFERRED
        want = (wanted or self.PREFERRED).removeprefix("models/")
        if want in names:
            return want
        auto = self.pick_model(names)
        if auto is None:
            raise RuntimeError(f"model '{want}' not available ({self.api_version}) and no Flash/Pro model found; "
                               f"available: {', '.join(sorted(names)) or '-'}")
        print(f"LLM: model '{want}' not available on {self.api_version} -> using '{auto}' "
              f"(set GEMINI_MODEL or --model to choose; python ask.py --list-models)")
        return auto

    @classmethod
    def pick_model(cls, names: list[str]) -> str | None:
        """Newest gemini-*-flash (stable before preview), else newest pro."""
        def version(n):
            m = re.match(r"gemini-(\d+(?:\.\d+)*)", n)
            return tuple(int(x) for x in m.group(1).split(".")) if m else ()
        for kind in ("flash", "pro"):
            cands = [n for n in names if n.startswith("gemini-") and kind in n and not any(s in n for s in cls._SKIP)]
            if cands:
                return max(cands, key=lambda n: (version(n), "latest" in n, "preview" not in n and "exp" not in n,
                                                 -len(n)))
        return None

    def start(self, system, decls, user):
        config = {"system_instruction": system, "tools": [{"function_declarations": decls}],
                  "automatic_function_calling": {"disable": True}}
        tc = self.thinking_config()
        if tc:
            config["thinking_config"] = tc
        return {"config": config, "contents": [{"role": "user", "parts": [{"text": user}]}]}

    def generate(self, state):
        for i in range(self.retries):
            try:
                config = state["config"]
                if state.get("force_final"):   # last step: only final_answer is allowed (partial answer / abstain)
                    config = {**config, "tool_config": {"function_calling_config": {
                        "mode": "ANY", "allowed_function_names": ["final_answer"]}}}
                resp = self.client.models.generate_content(model=self.model, contents=state["contents"],
                                                           config=config)
                break
            except Exception as e:  # rate limit / overload: back off and retry
                msg = str(e)
                if "thinking_config" in state["config"] and ("INVALID_ARGUMENT" in msg or "400" in msg) \
                        and "think" in msg.lower():   # model does not support this thinking setting
                    print(f"LLM: thinking setting '{self.thinking}' not supported by {self.model}, using the default")
                    self.thinking = "default"
                    state["config"].pop("thinking_config")
                    continue
                if i + 1 < self.retries and any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                                                                    "500", "INTERNAL")):
                    time.sleep(min(60, 2 ** (i + 1)))
                    continue
                raise
        cand = resp.candidates[0] if resp.candidates else None
        if cand is not None and cand.content is not None:
            state["contents"].append(cand.content)   # keep as is (thought signatures)
        calls = [(c.name, dict(c.args or {})) for c in (resp.function_calls or [])]
        text = None
        if not calls and cand is not None and cand.content is not None:
            text = "".join(p.text for p in (cand.content.parts or []) if getattr(p, "text", None))
        u = resp.usage_metadata
        usage = {"prompt": getattr(u, "prompt_token_count", 0) or 0,
                 "output": getattr(u, "candidates_token_count", 0) or 0,
                 "thinking": getattr(u, "thoughts_token_count", 0) or 0} if u else {}
        return {"calls": calls, "text": text, "usage": usage}

    def thinking_config(self) -> dict | None:
        """Less thinking = much faster per step (the agent checks with tools instead of long reasoning).
        Gemini 2.x: thinking_budget (tokens); Gemini 3+: thinking_level."""
        t = self.thinking
        if t in ("default", "", None):
            return None
        if re.match(r"gemini-(1|2)\.", self.model):
            return {"thinking_budget": {"off": 0, "low": 512, "medium": 2048, "high": 8192}.get(t, 512)}
        return {"thinking_level": {"off": "minimal", "low": "low", "medium": "medium", "high": "high"}.get(t, "low")}

    def add_results(self, state, outputs):
        parts, extra = [], []
        for name, out in outputs:
            if name == "_system":
                extra.append({"text": out["note"]})
                continue
            out = dict(out)
            img = out.pop("_image", None)
            parts.append({"function_response": {"name": name, "response": out}})
            if img is not None:   # images go as normal parts after the function responses of this turn
                extra += [{"text": f"Image returned by view ({out.get('shown')}, crop_px {out.get('crop_px')}):"},
                          {"inline_data": {"mime_type": "image/png", "data": img}}]
        state["contents"].append({"role": "user", "parts": parts + extra})

    def list_models(self) -> list[str]:
        return [m.name for m in self.client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]

    def judge(self, question: str, reference: str, prediction: str) -> dict:
        """LLM-as-judge for free-text answers: correct / partial / wrong + short reason."""
        prompt = (f"Question: {question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{prediction}\n\n"
                  "Does the candidate convey the same facts as the reference (numbers within 5 %, same items)? "
                  "If the reference says the information is not available, the candidate is correct only if it also "
                  "says so. Reply as JSON: {\"verdict\": \"correct\"|\"partial\"|\"wrong\", \"reason\": \"...\"}")
        for i in range(self.retries):
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt,
                                                        config={"response_mime_type": "application/json"})
                return json.loads(r.text)
            except Exception as e:
                if i + 1 == self.retries:
                    return {"verdict": "error", "reason": f"{type(e).__name__}: {e}"}
                time.sleep(min(60, 2 ** (i + 1)))
