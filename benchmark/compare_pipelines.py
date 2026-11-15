"""Benchmark: current pipeline (pipeline/, the one the app uses) vs. the multi-model pipeline (benchmark/pipeline_v2).

Both get the same images and the same questions and are scored with the same rule (pipeline_v2/evaluate_campi.py,
dataset README): counts must be exact, areas within 2 %. Nothing of the running app is changed or imported into it.

  v1  pipeline/   PaddleOCR + CubiCasa5K layout -> plan graph -> rules (planqa/qa.py);
                  with --v1-agent also the Gemini agent (as in the app; costs API tokens)
  v2  benchmark/pipeline_v2/main.py   up to 7 vision models in parallel -> merged.campi.json -> field per question

    # from the project root, with the Python env of the pipeline (torch, paddleocr, onnxruntime):
    python benchmark/compare_pipelines.py --questions ../Junction/test_questions.jsonl --images ../Junction
    python benchmark/compare_pipelines.py --questions ../Junction/questions.jsonl --images ../Junction --sample 20
    python benchmark/compare_pipelines.py ... --v2-models layout yolo_general yolo_symbols     # CPU subset (default)
    python benchmark/compare_pipelines.py ... --v2-facts ~/Downloads/Junction/facts           # v2 ran elsewhere (GPU VM)
    python benchmark/compare_pipelines.py ... --v1-agent --v1-vision                           # v1 like the app

Outputs in --out (default benchmark/runs/<timestamp>/): report.md, summary.json, results.jsonl (one line per
question: both answers, verdicts, the v2 field and the model it came from), the staged input images, v1 graphs,
v2 facts and logs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
V1_DIR = PROJECT / "pipeline"
V2_DIR = HERE / "pipeline_v2"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CPU_MODELS = ["layout", "yolo_general", "yolo_symbols"]


def load_v2_scorer():
    """pipeline_v2/evaluate_campi.py by file path (it only uses the standard library)."""
    spec = importlib.util.spec_from_file_location("v2_evaluate_campi", V2_DIR / "evaluate_campi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_stem(path: Path) -> str:
    """Same rule as pipeline_v2/main.py (its facts are stored under this name)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)


def question_type(q: dict) -> str | None:
    t = q.get("type")
    if t in ("count", "area"):
        return t
    unit = (q.get("unit") or "").lower()
    if unit == "count":
        return "count"
    if unit in ("m2", "m²", "sqm"):
        return "area"
    return None   # free text etc.: neither pipeline answers it deterministically


def find_image(path: str, roots: list[Path], cache: dict) -> Path | None:
    if path in cache:
        return cache[path]
    p = Path(path)
    found = None
    cands = [p] + [r / p for r in roots] + [r / p.name for r in roots]
    for c in cands:
        if c.is_file() and c.suffix.lower() in IMAGE_EXTS:
            found = c
            break
    if found is None:   # same drawing in another image format (README: same stem = same drawing)
        for r in roots:
            for c in r.rglob(p.stem + ".*"):
                if c.suffix.lower() in IMAGE_EXTS:
                    found = c
                    break
            if found:
                break
    cache[path] = found
    return found


# ---------------------------------------------------------------------------------------------------- v1

def run_v1(plans: dict[str, dict], out: Path, args) -> dict[int, dict]:
    """Current pipeline in this process. Returns question index -> {value, intent, explanation, seconds}."""
    sys.path.insert(0, str(V1_DIR))
    from answer import answer_items, llm_answers, make_agent   # noqa: E402  (pipeline/answer.py)
    from planqa.ocr import load_image   # noqa: E402
    from planqa.pipeline import Pipeline   # noqa: E402

    min_conf = {"texts": args.min_conf_text, "objects": args.min_conf_symbols, "rooms": args.min_conf_rooms}
    weights = Path(args.v1_weights)
    if not weights.is_file():   # without it v1 silently runs without the layout model: not a fair comparison
        raise SystemExit(f"v1 layout weights not found: {weights} (copy model_best_val_loss_var.pkl there or "
                         f"pass --v1-weights; --skip-v1 to run v2 only)")
    t0 = time.perf_counter()
    pipe = Pipeline(min_conf=min_conf, weights=str(weights), ocr_model=args.ocr_model, tta=not args.no_tta)
    load_s = time.perf_counter() - t0
    agent = make_agent(args.model, args.max_steps, args.thinking, 1, args.v1_vision, True) if args.v1_agent else None
    res: dict[int, dict] = {}
    for stem, plan in plans.items():
        t = time.perf_counter()
        graph, _ = pipe.process(plan["staged"], out / "v1" / stem, report=False)
        items = [{"question": plan["questions"][i]["question"], "unit": plan["questions"][i].get("unit")}
                 for i in plan["qidx"]]
        recs = answer_items(graph, items)
        if agent is not None:
            llm_answers(recs, graph, agent, load_image(plan["staged"]) if args.v1_vision else None)
        sec = time.perf_counter() - t
        for i, rec in zip(plan["qidx"], recs):
            res[i] = {"value": rec.get("prediction"), "intent": rec.get("intent"),
                      "explanation": rec.get("explanation"), "fallback": rec.get("fallback"),
                      "rule_value": rec.get("rule_prediction", rec.get("prediction"))}
        plan["v1_seconds"] = round(sec, 1)
        print(f"[v1] {stem}: {len(recs)} questions in {sec:.1f}s", flush=True)
    res["_load_seconds"] = round(load_s, 1)   # type: ignore[index]
    return res


# ---------------------------------------------------------------------------------------------------- v2

def run_v2(input_dir: Path, out: Path, args) -> tuple[Path, float, int]:
    facts = out / "v2" / "facts"
    work = out / "v2" / "work"
    log = out / "v2" / "main.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.v2_python or sys.executable, "main.py", str(input_dir.resolve()), "--output", str(facts.resolve()),
           "--work-dir", str(work.resolve()), "--device", args.device, "--models", *args.v2_models, "--local"]
    env = dict(os.environ)
    v2_weights = V2_DIR / "models" / "model_best_val_loss_var.pkl"
    if not v2_weights.is_file() and Path(args.v1_weights).is_file():
        env.setdefault("CUBICASA_WEIGHTS", str(Path(args.v1_weights).resolve()))   # same layout weights as v1
    print(f"[v2] {' '.join(cmd)}  (cwd {V2_DIR})", flush=True)
    t = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=V2_DIR, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    sec = time.perf_counter() - t
    print(f"[v2] exit {rc} after {sec:.1f}s (log: {log})", flush=True)
    return facts, sec, rc


# ---------------------------------------------------------------------------------------------------- scoring

def verdict(scorer, q: dict, value) -> str:
    if value is None:
        return "null"
    try:
        return "correct" if scorer.correct(q, value) else "wrong"
    except (TypeError, ValueError):
        return "wrong"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True, help="jsonl: path, question, answer, unit[, type]")
    ap.add_argument("--images", type=Path, nargs="+", required=True, help="folders to find the question paths in")
    ap.add_argument("--sample", type=int, default=0, help="random sample of N plans (seed 0)")
    ap.add_argument("--plans", type=Path, help="text file with plan stems to keep (one per line)")
    ap.add_argument("--exclude-plans", type=Path, help="text file with plan stems to drop (e.g. v2's validation split)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-v1", action="store_true")
    ap.add_argument("--skip-v2", action="store_true")
    # v1
    ap.add_argument("--min-conf-text", type=float, default=0.65)
    ap.add_argument("--min-conf-symbols", type=float, default=0.65)
    ap.add_argument("--min-conf-rooms", type=float, default=0.65)
    ap.add_argument("--ocr-model", default="small", choices=["tiny", "small", "medium", "auto"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--v1-weights", default=str(V1_DIR / "models" / "model_best_val_loss_var.pkl"))
    ap.add_argument("--v1-agent", action="store_true", help="answer with the Gemini agent like the app (API costs)")
    ap.add_argument("--v1-vision", action="store_true")
    ap.add_argument("--model", default=None, help="Gemini model for --v1-agent (default $GEMINI_MODEL)")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--thinking", default=None)
    # v2
    ap.add_argument("--v2-models", nargs="+", default=CPU_MODELS,
                    help="pipeline_v2 models (default: the CPU subset; all 7 need CUDA + their venvs)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v2-python", default=None, help="interpreter for pipeline_v2/main.py (default: this one)")
    ap.add_argument("--v2-facts", type=Path, help="existing main.py --output folder: score it instead of running v2")
    ap.add_argument("--v2-file", default="merged.campi.json", help="facts file per plan (e.g. layout.campi.json)")
    args = ap.parse_args()

    scorer = load_v2_scorer()
    out = args.out or HERE / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ questions -> plans
    questions = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = set(args.plans.read_text().split()) if args.plans else None
    drop = set(args.exclude_plans.read_text().split()) if args.exclude_plans else set()
    cache: dict = {}
    plans: dict[str, dict] = {}
    skipped = Counter()
    for i, q in enumerate(questions):
        qt = question_type(q)
        if qt is None:
            skipped["no count/area type"] += 1
            continue
        q["type"] = qt
        img = find_image(q["path"], args.images, cache)
        if img is None:
            skipped["image not found"] += 1
            continue
        stem = safe_stem(img)
        if (keep is not None and stem not in keep) or stem in drop:
            skipped["filtered by --plans/--exclude-plans"] += 1
            continue
        plan = plans.setdefault(stem, {"image": img, "qidx": [], "questions": questions})
        plan["qidx"].append(i)
    if args.sample and len(plans) > args.sample:
        chosen = sorted(random.Random(0).sample(sorted(plans), args.sample))
        plans = {k: plans[k] for k in chosen}
    if not plans:
        raise SystemExit(f"no plans to run ({dict(skipped)})")
    n_q = sum(len(p["qidx"]) for p in plans.values())
    print(f"{n_q} questions on {len(plans)} plans" + (f"; skipped: {dict(skipped)}" if skipped else ""), flush=True)

    input_dir = out / "input"   # one folder holding only the images (pipeline_v2 reads every file in it)
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)
    for stem, plan in plans.items():
        plan["staged"] = input_dir / f"{stem}{plan['image'].suffix.lower()}"
        shutil.copy2(plan["image"], plan["staged"])

    # ------------------------------------------------------------------ run
    v1 = {} if args.skip_v1 else run_v1(plans, out, args)
    v2_run = None
    if args.skip_v2:
        facts_dir = None
    elif args.v2_facts:
        facts_dir = args.v2_facts
    else:
        facts_dir, sec, rc = run_v2(input_dir, out, args)
        v2_run = {"seconds": round(sec, 1), "returncode": rc, "models": args.v2_models, "device": args.device}

    # ------------------------------------------------------------------ score
    rows = []
    facts_cache: dict[str, dict | None] = {}
    for stem, plan in plans.items():
        doc = None
        if facts_dir is not None:
            if stem not in facts_cache:
                f = facts_dir / stem / args.v2_file
                facts_cache[stem] = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
            doc = facts_cache[stem]
        for i in plan["qidx"]:
            q = questions[i]
            kind, _, field = scorer.predict(q, {})   # question kind independent of the facts
            row = {"plan": stem, "path": q["path"], "question": q["question"], "type": q["type"],
                   "expected": q["answer"], "kind": kind}
            if not args.skip_v1:
                a = v1.get(i, {})
                row.update(v1_value=a.get("value"), v1_intent=a.get("intent"),
                           v1_verdict=verdict(scorer, q, a.get("value")), v1_explanation=a.get("explanation"))
                if args.v1_agent:
                    row["v1_rule_value"] = a.get("rule_value")
            if facts_dir is not None:
                if doc is None:
                    row.update(v2_value=None, v2_verdict="null", v2_field=None, v2_model=None, v2_note="no facts")
                else:
                    _, value, field = scorer.predict(q, doc.get("floor_plan") or {})
                    model = ((doc.get("_meta") or {}).get("sources") or {}).get(field) if field else None
                    row.update(v2_value=value, v2_verdict=verdict(scorer, q, value), v2_field=field, v2_model=model,
                               v2_uncertain=field in ((doc.get("_meta") or {}).get("uncertain_fields") or []))
            rows.append(row)
    # which v2 models actually produced output (main.py exits 0 even when a model failed)
    v2_models_seen, v2_failed = Counter(), set()
    for doc in facts_cache.values():
        if doc:
            v2_models_seen.update((doc.get("_meta") or {}).get("models") or [])
            for name, run in ((doc.get("_meta") or {}).get("runs") or {}).items():
                if not run.get("ok", True):
                    v2_failed.add(name)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------ report
    sides = [s for s in ("v1", "v2") if any(f"{s}_verdict" in r for r in rows)]
    by_kind: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    pair = Counter()
    for r in rows:
        for s in sides:
            by_kind[r["kind"]][s][r[f"{s}_verdict"]] += 1
            by_kind["ALL"][s][r[f"{s}_verdict"]] += 1
        if len(sides) == 2:
            a, b = r["v1_verdict"] == "correct", r["v2_verdict"] == "correct"
            pair["both" if a and b else "only_v1" if a else "only_v2" if b else "neither"] += 1

    def acc(c: Counter) -> str:
        n = sum(c.values())
        return f"{c['correct'] / n:.1%}" if n else "-"

    lines = [f"# Pipeline benchmark – {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Questions: `{args.questions}` · {n_q} questions on {len(plans)} plans"
             + (f" · skipped: {dict(skipped)}" if skipped else ""),
             "Scoring: counts exact, areas within 2 % (pipeline_v2/evaluate_campi.py); `null` = no answer.", ""]
    if "v1" in sides:
        lines.append(f"- **v1** (app pipeline): thresholds T/S/R {args.min_conf_text}/{args.min_conf_symbols}/"
                     f"{args.min_conf_rooms}, " + ("Gemini agent" + (" + vision" if args.v1_vision else "") if args.v1_agent
                                                   else "rules only") + f", model load {v1.get('_load_seconds')} s")
    if "v2" in sides:
        lines.append(f"- **v2** (multi-model): " + (f"models {', '.join(args.v2_models)} on {args.device}, "
                                                    f"{v2_run['seconds']} s total, exit {v2_run['returncode']}"
                                                    if v2_run else f"facts from `{facts_dir}`")
                     + f", file `{args.v2_file}`")
    if "v2" in sides:
        n_docs = sum(1 for d in facts_cache.values() if d)
        lines.append(f"- v2 models with output: " + (", ".join(f"{m} ({c}/{n_docs} plans)" for m, c in
                                                            sorted(v2_models_seen.items())) or "**none**"))
        missing = sorted(set(args.v2_models) - set(v2_models_seen)) if not args.v2_facts else []
        if missing or v2_failed or not v2_models_seen:
            lines.append(f"- **WARNING: v2 models failed or produced nothing: {', '.join(sorted(set(missing) | v2_failed)) or 'all'}"
                         f" – see `{out / 'v2' / 'main.log'}` and the model logs. Their questions count as null.**")
    lines += ["", "| question kind | n | " + " | ".join(f"{s} correct | {s} wrong | {s} null | {s} acc" for s in sides)
              + " |", "|---|---|" + "---|---|---|---|" * len(sides)]
    for kind in sorted(k for k in by_kind if k != "ALL") + ["ALL"]:
        cs = by_kind[kind]
        n = sum(cs[sides[0]].values())
        name = f"**{kind}**" if kind == "ALL" else kind
        lines.append(f"| {name} | {n} | " + " | ".join(
            f"{cs[s]['correct']} | {cs[s]['wrong']} | {cs[s]['null']} | {acc(cs[s])}" for s in sides) + " |")
    if pair:
        lines += ["", f"Per question: both correct {pair['both']}, only v1 {pair['only_v1']}, "
                      f"only v2 {pair['only_v2']}, neither {pair['neither']}."]
    if "v1" in sides:
        secs = [p.get("v1_seconds") for p in plans.values() if p.get("v1_seconds") is not None]
        if secs:
            lines.append(f"v1 time per plan: {sum(secs) / len(secs):.1f} s mean (excl. model load).")
    if v2_run:
        lines.append(f"v2 wall time per plan: {v2_run['seconds'] / len(plans):.1f} s (incl. model loading, all models "
                     f"in parallel). Model logs: `{out / 'v2' / 'work' / 'logs'}`.")
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    summary = {"questions": str(args.questions), "n_questions": n_q, "n_plans": len(plans), "skipped": dict(skipped),
               "by_kind": {k: {s: dict(c) for s, c in v.items()} for k, v in by_kind.items()},
               "pairs": dict(pair), "v2_run": v2_run, "v1_load_seconds": v1.get("_load_seconds") if v1 else None,
               "args": {k: str(v) for k, v in vars(args).items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + report)
    print(f"-> {out / 'report.md'}, {out / 'results.jsonl'}")


if __name__ == "__main__":
    main()
