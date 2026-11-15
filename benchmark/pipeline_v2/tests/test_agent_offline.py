"""Offline test of the agent: JSON tools, agent loop (scripted model) and the Gemini backend
against a fake google.genai (no network, no API key).

    python tests/test_agent_offline.py
"""
import json
import sys
import types as pytypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_ifc_offline import model

from planqa.agent import Agent, GeminiBackend, JsonTools, ScriptedBackend, _safe_eval
from planqa.ifc_extract import build_graph

doc = json.loads(build_graph(model()).model_dump_json(exclude_none=True))
T = JsonTools(doc)

# ---- tools
assert T.kind == "ifc" and {"spaces", "elements.walls", "elements.doors"} <= set(T.tables), T.tables
ov = T.overview()
assert ov["storeys"][0]["name"] == "EG" and ov["tables"]["elements.walls"]["rows"] == 6
q = T.query_table("spaces", where=[{"column": "name", "op": "contains", "value": "bad"},
                                   {"column": "storey", "op": "=", "value": "L0"}], columns=["id", "name", "net_area"])
assert q["rows"] == [["S2", "Bad", 11.2]], q
assert T.aggregate("elements.walls", "sum", "length", where=[{"column": "storey", "op": "=", "value": "L0"}])["value"] \
    == round(2 * 8.73 + 2 * 4 + 4, 4)
g = T.aggregate("elements.walls", "count", group_by="ext")
assert g["groups"] == {"False": 1, "True": 5}, g
assert T.aggregate("elements.doors", "max", "width")["value"] == 1.01
assert T.query_table("elements.walls", where=[{"column": "ext", "op": "=", "value": "true"},
                                              {"column": "length", "op": ">", "value": "5"}])["n_matched"] == 3
assert T.query_table("spaces", where=[{"column": "walls", "op": "contains", "value": "W5"}])["n_matched"] == 2
top = T.query_table("spaces", sort_by="net_area", descending=True, limit=1)
assert dict(zip(top["columns"], top["rows"][0]))["name"] == "Schlafen" and "guid" not in top["columns"]   # heavy cols hidden
assert "guid" in T.query_table("spaces", columns=["id", "guid"])["columns"]
# conditions sent as strings / JSON strings by the model are understood
from planqa.agent import _call_tool

assert T.query_table("spaces", where=["storey = L0"])["n_matched"] == 2
assert T.query_table("spaces", where=['{"column": "name", "op": "contains", "value": "bad"}'])["n_matched"] == 1
assert T.aggregate("elements.walls", "count", where="ext = true")["value"] == 5
assert "not understood" in _call_tool(T, "query_table", {"table": "spaces", "where": ["???"]})["error"]
assert "error" in _call_tool(T, "query_table", {"table": "spaces", "limit": "x"})        # errors go back to the model
T2 = JsonTools(doc)
T2.calculate = lambda expression: "x".get("y")                                        # AttributeError like in the run
assert "failed: AttributeError" in _call_tool(T2, "calculate", {"expression": "1"})["error"]
s = T.search("schlafen")
assert any(h.get("row_id") == "S3" for h in s["hits"]), s
assert any(h.get("table") == "storeys" and h["row_id"] == "L1" for h in T.search("OG")["hits"])
assert any(h.get("path") == "building.address" for h in T.search("wien")["hits"])
assert T.get("summary.doors.width_max") == 1.01
assert _safe_eval("round(8.73*2+4*2+4, 2)") == 29.46 and _safe_eval("sum([1,2,3])") == 6
for bad in ("__import__('os')", "open('x')", "(1).real"):
    try:
        _safe_eval(bad)
        raise AssertionError(bad)
    except ValueError:
        pass
# errors come back as tool results, not exceptions
from planqa.agent import _call_tool

assert "unknown column" in _call_tool(T, "query_table", {"table": "spaces", "where": [{"column": "nope", "op": "="}]})["error"]
assert "not found" in _call_tool(T, "get", {"path": "nope.x"})["error"]
assert "unknown tool" in _call_tool(T, "rm", {})["error"]

# ---- agent loop with a scripted model
script = [
    [("overview", {})],
    [("search", {"text": "Bad"}), ("query_table", {"table": "spaces", "where": [{"column": "id", "op": "=", "value": "S2"}]})],
    [("final_answer", {"answer": "Das Bad hat 11,2 m².", "value": 11.2, "unit": "m2", "evidence": ["S2"],
                       "confidence": 0.95, "reason": "spaces.net_area of S2"})],
]
be = ScriptedBackend(script)
r = Agent(be).ask(doc, "Wie groß ist das Bad?")
assert r.value == 11.2 and r.evidence == ["S2"] and r.steps == 3 and not r.error, r
assert [t["tool"] for t in r.trace] == ["overview", "overview", "search", "query_table", "final_answer"]   # 1st = in prompt
assert len(be.seen) == 2 and be.seen[1][1][1]["n_matched"] == 1
class Spy(ScriptedBackend):
    def generate(self, state):
        self.forced = state.get("force_final")
        return super().generate(state)
spy = Spy([[("overview", {})]] * 20)
r = Agent(spy, max_steps=4).ask(doc, "loop")
assert r.error and "step limit" in r.error and spy.forced, "last step must force final_answer"

# provenance and assumptions of an estimate are kept
r = Agent(ScriptedBackend([[("final_answer", {"answer": "Not directly in the model: about 4 knobs (2 doors x 2).",
                                               "value": 4, "confidence": 0.5, "reason": "estimate", "provenance": "estimate",
                                               "assumptions": ["2 knobs per door"]})]])).ask(doc, "doorknobs?")
assert r.provenance == "estimate" and r.assumptions == ["2 knobs per door"] and not r.abstained

# ---- Gemini backend against a fake google.genai
calls_seen = []


class FC:
    def __init__(self, name, args):
        self.name, self.args = name, args


class Resp:
    def __init__(self, calls=None, text=None):
        parts = [pytypes.SimpleNamespace(text=text)] if text else []
        self.candidates = [pytypes.SimpleNamespace(content=pytypes.SimpleNamespace(role="model", parts=parts))]
        self.function_calls = [FC(*c) for c in calls] if calls else None
        self.usage_metadata = pytypes.SimpleNamespace(prompt_token_count=100, candidates_token_count=10,
                                                      thoughts_token_count=5)
        self.text = text


class Models:
    def __init__(self):
        self.n = 0

    def generate_content(self, model, contents, config):
        calls_seen.append((model, json.loads(json.dumps(contents[-1], default=lambda o: o.__dict__)), config))
        self.n += 1
        if self.n == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")    # retried
        if self.n == 2:
            return Resp([("aggregate", {"table": "elements.doors", "op": "count",
                                        "where": [{"column": "storey", "op": "=", "value": "L0"}]})])
        return Resp([("final_answer", {"answer": "2 doors on EG", "value": 2, "unit": "count", "evidence": ["L0"],
                                       "confidence": 0.9, "reason": "count of elements.doors with storey L0"})])

    def list(self):
        return [pytypes.SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
                for n in ("gemini-test", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview")]


fake = pytypes.ModuleType("google.genai")
client_opts = []
fake.Client = lambda http_options=None: client_opts.append(http_options) or pytypes.SimpleNamespace(models=Models())
google = sys.modules.setdefault("google", pytypes.ModuleType("google"))
google.genai = fake
sys.modules["google.genai"] = fake
import planqa.agent as ag

ag.time.sleep = lambda s: None
gb = GeminiBackend("gemini-test")
r = Agent(gb).ask(doc, "How many doors are on EG?")
assert r.value == 2 and r.model == "gemini-test" and r.tokens["prompt"] == 200 and not r.error, r
assert client_opts[0] == {"api_version": "v1beta"}, client_opts
assert calls_seen[0][2]["thinking_config"] == {"thinking_level": "low"}, calls_seen[0][2]
assert "tool_config" not in calls_seen[0][2]
assert GeminiBackend("gemini-2.5-flash", thinking="off").thinking_config() == {"thinking_budget": 0}
assert GeminiBackend("gemini-test", thinking="default").thinking_config() is None
assert "Document overview" in calls_seen[0][1]["parts"][0]["text"]   # overview sent with the question
assert GeminiBackend("gemini-flash-latest").model == "gemini-3-flash-preview"   # missing alias -> newest Flash
cfg = calls_seen[0][2]
assert cfg["automatic_function_calling"] == {"disable": True} and "final_answer" in \
    [d["name"] for d in cfg["tools"][0]["function_declarations"]]
fr = calls_seen[-1][1]["parts"][0]["function_response"]
assert fr["name"] == "aggregate" and fr["response"]["value"] == 2, fr
print("agent: all checks passed")
