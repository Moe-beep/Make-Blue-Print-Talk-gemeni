"""Answer questions about plans, with evidence, and evaluate against ground truth.

Benchmark (default): the 4 images and their questions in test_questions.jsonl
    python answer.py

Single question:
    python answer.py --image data/high_quality-1500.png --question "How many windows are shown on this floor plan?"

Question file (jsonl with path, question[, answer]); images are found by file name in --images:
    python answer.py --questions questions.jsonl --images data
    python answer.py --questions questions.jsonl --images data --sample 10

LLM agent (default, needs `pip install google-genai` and GEMINI_API_KEY): every question is
answered by Gemini from the generated plan_graph.json via tools (planqa/agent.py). The
rule-based answer (planqa/qa.py) is given to the agent as an estimate to verify, is used as
fallback (no key, API error), and is reported side by side. --no-llm = rules only.
    python answer.py --min-conf-text 0.65 --min-conf-symbols 0.65 --min-conf-rooms 0.65
    python answer.py --model gemini-pro-latest
    python answer.py --no-llm
    python answer.py --vision --llm-workers 4      # agent may look at crops of the drawing (view tool)
    python answer.py --vision --free-overrides     # without the check that deviations are itemized

OCR + layout always run fresh. --sweep additionally compares a grid of text/symbol thresholds
on the detections of this run (rebuilding the graph takes milliseconds, no extra model calls):
    python answer.py --min-conf-text 0.5 --min-conf-symbols 0.5
    python answer.py --sweep
Writes out/<plan>/answers.json + report.html (click an answer -> its evidence),
out/predictions.jsonl, accuracy per question type vs. the fallback baseline and the
signed error (too many / too few) per type.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from planqa.qa import PRIOR, answer, route

_PIPE = None  # models are loaded once and reused for all images


def make_agent(model: str | None, max_steps: int, thinking: str | None = None, workers: int = 1, vision: bool = False,
               check_overrides: bool = True):
    """Gemini agent, or None (with the reason) if the SDK or the API key is missing."""
    import os
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("LLM: GEMINI_API_KEY not set -> rule-based answers only (export GEMINI_API_KEY=...)")
        return None
    try:
        from planqa.agent import Agent, GeminiBackend
        agent = Agent(GeminiBackend(model, thinking=thinking), max_steps)
        agent.workers, agent.vision, agent.check_overrides = max(1, workers), vision, check_overrides
    except ImportError:
        print("LLM: google-genai not installed -> rule-based answers only (pip install google-genai)")
        return None
    except Exception as e:
        print(f"LLM: Gemini client failed ({type(e).__name__}: {e}) -> rule-based answers only")
        return None
    print(f"LLM: {agent.backend.model} (thinking {agent.backend.thinking}, {agent.workers} parallel question(s)"
          f"{', view tool on the drawing' if vision else ''}"
          f"{', deviations must be itemized' if check_overrides else ', free overrides'}) "
          "answers from building_graph (rules = estimate + fallback)")
    return agent


def find_image(path: str, roots: list[Path]) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    for root in roots:
        for cand in (root / p, root / p.name):
            if cand.is_file():
                return cand
    return None


def correct(pred, truth, unit) -> bool:
    if pred is None or truth is None:
        return False
    if unit == "count":
        return int(round(pred)) == int(truth)
    return abs(pred - truth) <= max(0.05 * abs(truth), 0.1)  # areas etc.: 5 % tolerance


def process_image(img: Path, items: list[dict], out_root: str, cfg: dict, agent=None) -> tuple[list[dict], dict]:
    """Run the models on one image, answer, write report. Returns (records, raw detections)."""
    global _PIPE
    from planqa.ocr import load_image
    from planqa.pipeline import Pipeline
    from planqa.visualize import html_report

    t = time.perf_counter()
    out_dir = Path(out_root) / img.stem
    if _PIPE is None:
        _PIPE = Pipeline(**cfg)
    graph, _ = _PIPE.process(img, out_dir, report=False)
    recs = answer_items(graph, items)
    if agent is not None:
        llm_answers(recs, graph, agent, load_image(img) if getattr(agent, "vision", False) else None)
    (out_dir / "answers.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    html_report(load_image(img), graph, out_dir / "report.html", layout_img=_PIPE.last_layout_img, answers=recs)
    try:   # explanation.json + uncertainty_*.png + explain.html (evidence, uncertainty map, agent coverage)
        from planqa.explain import build_explanation, write_explanation
        models = {"ocr": f"PP-OCRv6_{cfg.get('ocr_model', 'small')}", "layout": "cubicasa5k"}
        if agent is not None:
            models["agent"] = agent.backend.model
        doc, heat = build_explanation(_PIPE.last_image, graph, _PIPE.last_detections, getattr(_PIPE, "last_probs", None),
                                      recs, cfg.get("min_conf"), models)
        write_explanation(doc, heat, _PIPE.last_image, out_dir)
    except Exception as e:   # explanations must never break the benchmark
        print(f"   explanation failed: {type(e).__name__}: {e}")
    print(f"{img.name}: {len(recs)} questions in {time.perf_counter() - t:.1f}s -> {out_dir / 'explain.html'}", flush=True)
    return recs, _PIPE.last_detections


def answer_items(graph, items: list[dict]) -> list[dict]:
    from planqa.plan_to_graph import ROOM_INTENTS, room_answer, to_building_graph
    bg = to_building_graph(graph)   # room counts: same room assignment as facts (labels, OCR typos, split rooms)
    recs = []
    for it in items:
        intent = route(it["question"])
        a = answer(graph, intent)
        if intent in ROOM_INTENTS and not a.fallback:
            a = room_answer(bg, intent)
        rec = {**it, "intent": intent, "prediction": a.value, "confidence": a.confidence,
               "fallback": a.fallback, "evidence": a.evidence, "explanation": a.explanation}
        if "answer" in it and intent:
            rec["correct"] = correct(a.value, it["answer"], it.get("unit", "count"))
        recs.append(rec)
    return recs


def llm_answers(recs: list[dict], graph, agent, image=None) -> None:
    """Replace the rule answers by the agent's answers (rule answer kept in rule_* fields)."""
    from planqa.plan_to_graph import to_building_graph
    doc = json.loads(to_building_graph(graph).model_dump_json(exclude_none=True))   # common schema, same as IFC
    def ctx_of(rec):
        return (f"Rule-based estimate: {rec['prediction']} ({rec['explanation']} Evidence: "
                f"{', '.join(rec['evidence']) or '-'})") if rec["intent"] else None

    def estimate_of(rec):   # count questions: the agent may only deviate item by item (see Agent._check_override)
        ok = getattr(agent, "check_overrides", True) and rec["intent"] and not rec["fallback"] \
            and rec.get("unit", "count") == "count"
        return rec["prediction"] if ok and rec["prediction"] is not None else None

    jobs = [(rec["question"], ctx_of(rec), estimate_of(rec)) for rec in recs]
    workers = getattr(agent, "workers", 1)
    if workers > 1:   # questions are independent; the time is spent waiting for the API
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            results = iter(list(ex.map(lambda j: agent.ask(doc, j[0], j[1], image=image, estimate=j[2]), jobs)))
    else:
        results = (agent.ask(doc, j[0], j[1], image=image, estimate=j[2]) for j in jobs)
    for rec, r in zip(recs, results):
        rule = {k: rec[k] for k in ("prediction", "confidence", "evidence", "explanation", "fallback")}
        use_llm = r.value is not None and not r.error
        rec.update({f"rule_{k}": v for k, v in rule.items()})
        if "correct" in rec:
            rec["rule_correct"] = rec["correct"]
        rec.update(llm_changes=r.changes, llm_override=r.override, llm_provenance=r.provenance,
                   llm_assumptions=r.assumptions)
        rec.update(llm_answer=r.answer, llm_reason=r.reason, abstained=r.abstained, llm_error=r.error,
                   steps=r.steps, seconds=r.seconds, tokens=r.tokens, trace=r.trace, llm_fallback=not use_llm)
        if use_llm:
            rec.update(prediction=r.value, confidence=round(r.confidence, 2), evidence=r.evidence, fallback=False,
                       explanation=f"LLM ({r.model}): {r.answer} – {r.reason}")
        else:
            rec["explanation"] = f"LLM without result ({r.error or 'no numeric value'}), rule answer: {rule['explanation']}"
        if "answer" in rec and rec["intent"]:
            rec["correct"] = correct(rec["prediction"], rec["answer"], rec.get("unit", "count"))
        mark = "✓" if rec.get("correct") else "✗" if "correct" in rec else " "
        print(f"   {mark} {rec['question'][:60]:<60} llm={r.value} rules={rule['prediction']}"
              f"{' truth=' + str(rec['answer']) if 'answer' in rec else ''}"
              f"{'  FALLBACK ' + (r.error or 'no value') if not use_llm else ''}"
              f"{'  OVERRIDE ' + r.override[:70] if r.override else ''}  [{r.steps} steps, {r.seconds:.0f}s]",
              flush=True)


def error_stats(preds: list[dict]) -> dict:
    """Per intent: n, correct, baseline correct, signed errors (prediction - truth)."""
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "rule": 0, "base": 0, "err": []})
    for p in preds:
        if "correct" not in p:
            continue
        s = stats[p["intent"]]
        s["n"] += 1
        s["ok"] += p["correct"]
        s["rule"] += p.get("rule_correct", p["correct"])
        s["base"] += correct(PRIOR.get(p["intent"]), p["answer"], p.get("unit", "count"))
        if p["prediction"] is not None and isinstance(p["answer"], (int, float)):
            s["err"].append(p["prediction"] - p["answer"])
    return stats


def sweep(dets: dict, by_image: dict, room_conf: float) -> None:
    """Accuracy for a grid of text/symbol thresholds on the detections of this run (no model calls)."""
    from planqa.pipeline import graph_from_detections

    if not dets:
        return
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for tt in grid:
        for ts in grid:
            th = {"texts": tt, "objects": ts, "rooms": room_conf}
            preds = [r for img, d in dets.items() for r in answer_items(graph_from_detections(d, th), by_image[img])]
            st = error_stats(preds)
            n = sum(s["n"] for s in st.values())
            errs = [e for s in st.values() for e in s["err"]]
            rows.append((sum(s["ok"] for s in st.values()) / n, tt, ts,
                         sum(abs(e) for e in errs) / len(errs), sum(e > 0 for e in errs), sum(e < 0 for e in errs)))
    print(f"\nthreshold sweep on {len(dets)} image(s), rooms >= {room_conf} (best first)")
    print(f"{'texts':>6} {'symbols':>8} {'accuracy':>9} {'MAE':>6} {'too many':>9} {'too few':>8}")
    for acc, tt, ts, mae, over, under in sorted(rows, key=lambda r: (-r[0], r[3]))[:15]:
        print(f"{tt:6.1f} {ts:8.1f} {acc:9.1%} {mae:6.2f} {over:9d} {under:8d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--question")
    ap.add_argument("--questions", default="test_questions.jsonl",
                    help="jsonl with path, question[, answer, unit]; default = the benchmark")
    ap.add_argument("--images", nargs="*", default=["data"], help="folders to look up image file names")
    ap.add_argument("--sample", type=int, default=0, help="random sample of N images (fixed seed)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--ocr-model", default="small", choices=["tiny", "small", "medium", "auto"])
    ap.add_argument("--ocr-scale", type=int, default=1600)
    ap.add_argument("--no-tta", action="store_true", help="layout without 4x rotation averaging (4x faster)")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="one threshold for texts, rooms and symbols (overrides the three below)")
    ap.add_argument("--min-conf-text", type=float, default=0.6, help="drop OCR texts below this confidence")
    ap.add_argument("--min-conf-symbols", type=float, default=0.6, help="drop symbols (doors, windows, ...) below")
    ap.add_argument("--min-conf-rooms", type=float, default=0.8, help="drop rooms whose class confidence is below")
    ap.add_argument("--sweep", action="store_true",
                    help="also compare a grid of text/symbol thresholds on this run's detections")
    ap.add_argument("--weights", default="models/model_best_val_loss_var.pkl")
    ap.add_argument("--no-llm", action="store_true", help="rule-based answers only (no Gemini)")
    ap.add_argument("--model", default=None, help="Gemini model id (default $GEMINI_MODEL, else gemini-flash-latest if available, else newest Flash)")
    ap.add_argument("--max-steps", type=int, default=10, help="max tool rounds of the agent per question")
    ap.add_argument("--thinking", default=None, choices=["off", "low", "medium", "high", "default"],
                    help="Gemini thinking effort (default $GEMINI_THINKING or low; less = faster)")
    ap.add_argument("--llm-workers", type=int, default=1, help="questions of one image sent to Gemini in parallel")
    ap.add_argument("--vision", action="store_true", help="agent may look at crops of the drawing (view tool)")
    ap.add_argument("--free-overrides", action="store_true",
                    help="accept any LLM count; default: a count != estimate needs itemized `changes`")
    args = ap.parse_args()

    if args.image:
        items = [{"path": args.image, "question": args.question or ""}]
    else:
        items = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    by_image, missing = defaultdict(list), set()
    for it in items:
        img = find_image(it["path"], [Path(r) for r in args.images])
        if img is None or img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            missing.add(it["path"])
        else:
            by_image[img].append(it)
    images = sorted(by_image)
    if args.sample:
        images = sorted(random.Random(0).sample(images, min(args.sample, len(images))))
    print(f"{sum(len(by_image[i]) for i in images)} questions on {len(images)} images: "
          + ", ".join(i.name for i in images))
    if missing:
        print(f"NOT FOUND (put them into one of {args.images} or pass their folder with --images): "
              + ", ".join(sorted({Path(m).name for m in missing})))

    min_conf = args.min_conf if args.min_conf is not None else \
        {"texts": args.min_conf_text, "objects": args.min_conf_symbols, "rooms": args.min_conf_rooms}
    print("thresholds:", min_conf)
    cfg = dict(tta=not args.no_tta, weights=args.weights, ocr_model=args.ocr_model, ocr_scale=args.ocr_scale,
               min_conf=min_conf)
    agent = None if args.no_llm else make_agent(args.model, args.max_steps, args.thinking, args.llm_workers, args.vision,
                                                not args.free_overrides)
    t0 = time.perf_counter()
    preds, dets = [], {}
    for img in images:
        recs, dets[img] = process_image(img, by_image[img], args.out, cfg, agent)
        preds += recs
    print(f"total {time.perf_counter() - t0:.0f}s for {len(images)} images")

    Path(args.out).mkdir(exist_ok=True)
    with open(Path(args.out) / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    if args.image:
        for p in preds:
            print(f"\nQ: {p['question']}\nA: {p['prediction']}  (confidence {p['confidence']:.2f}"
                  f"{', fallback' if p['fallback'] else ''})\n   {p['explanation']}\n   evidence: {', '.join(p['evidence']) or '-'}")
            for t in p.get("trace", []):
                print(f"     -> {t['tool']}({json.dumps(t['args'], ensure_ascii=False)[:120]})")
            if "rule_prediction" in p:
                print(f"   rules: {p['rule_prediction']} ({p['rule_explanation']})")
    stats = error_stats(preds)
    if stats:
        llm = any("rule_prediction" in p for p in preds)
        print(f"\n{'question type':<18} {'n':>3} {'accuracy':>9} {'rules':>7} {'baseline':>9} {'mean err':>9} {'MAE':>6} "
              f"{'too many':>9} {'too few':>8}" + ("   (accuracy = LLM agent)" if llm else ""))
        tot = {"n": 0, "ok": 0, "rule": 0, "base": 0, "err": []}
        for intent, s in sorted(stats.items()):
            e = s["err"]
            print(f"{intent:<18} {s['n']:3d} {s['ok'] / s['n']:9.1%} {s['rule'] / s['n']:7.1%} {s['base'] / s['n']:9.1%} "
                  f"{(sum(e) / len(e) if e else 0):+9.2f} {(sum(map(abs, e)) / len(e) if e else 0):6.2f} "
                  f"{sum(x > 0 for x in e):9d} {sum(x < 0 for x in e):8d}")
            for k in ("n", "ok", "rule", "base"):
                tot[k] += s[k]
            tot["err"] += e
        e = tot["err"]
        print(f"{'total':<18} {tot['n']:3d} {tot['ok'] / tot['n']:9.1%} {tot['rule'] / tot['n']:7.1%} {tot['base'] / tot['n']:9.1%} "
              f"{(sum(e) / len(e) if e else 0):+9.2f} {(sum(map(abs, e)) / len(e) if e else 0):6.2f} "
              f"{sum(x > 0 for x in e):9d} {sum(x < 0 for x in e):8d}")
        print("mean err = prediction - truth (negative: too few counted); MAE = mean |error|")
        if llm:
            tok = defaultdict(int)
            for p in preds:
                for k, v in (p.get("tokens") or {}).items():
                    tok[k] += v
            print(f"LLM: {sum(p.get('llm_fallback', False) for p in preds)} fallback(s) to rules, "
                  f"{sum(p.get('abstained', False) for p in preds)} abstained, "
                  f"avg {sum(p.get('steps', 0) for p in preds) / len(preds):.1f} steps, tokens {dict(tok)}")
        per_img = defaultdict(lambda: [0, 0])
        for p in preds:
            if "correct" in p:
                k = (Path(p["path"]).name, p.get("difficulty", ""))
                per_img[k][0] += 1; per_img[k][1] += p["correct"]
        print(f"\n{'image':<40} {'difficulty':<12} {'n':>3} {'accuracy':>9}")
        for (name, diff), (n, ok) in sorted(per_img.items(), key=lambda x: x[1][1] / x[1][0], reverse=True):
            print(f"{name:<40} {diff:<12} {n:3d} {ok / n:9.1%}")
    if args.sweep:   # rule-based only (LLM calls for 49 settings would be too slow and costly)
        sweep(dets, by_image, args.min_conf_rooms if args.min_conf is None else args.min_conf)
    print(f"-> {Path(args.out) / 'predictions.jsonl'}, reports in {args.out}/<plan>/report.html")


if __name__ == "__main__":
    main()
