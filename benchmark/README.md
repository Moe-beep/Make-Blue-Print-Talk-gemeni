# Pipeline benchmark: app pipeline (v1) vs. multi-model pipeline (v2)

Runs both pipelines on the same plans and questions and scores them with the same rule. The app is not touched:
`benchmark/` is not imported by the app and is excluded from the Docker image (`.dockerignore`).

| | v1 – `pipeline/` (what the app uses) | v2 – `benchmark/pipeline_v2/` (copy of `~/Downloads/Junction`) |
|---|---|---|
| Vision | PaddleOCR + CubiCasa5K layout | up to 7 models in parallel + vector-PDF reader, merged per field |
| Answer | rules (`planqa/qa.py`); `--v1-agent`: Gemini agent like the app | field of `merged.campi.json` (`evaluate_campi.predict`) |
| Runs | in the benchmark process | `pipeline_v2/main.py` as a subprocess (own `planqa/`, no clash) |

Scoring (`pipeline_v2/evaluate_campi.py`, dataset README): counts exact, areas within 2 %. A question without an
answer counts as `null` (not correct). Free-text questions are skipped – neither pipeline answers them
deterministically.

## Setup (once)

```bash
cd ~/PycharmProjects/Make_Blueprint_Talk
source ../Junction/.venv/bin/activate        # the env with torch (CPU), paddleocr, pydantic, opencv
bash benchmark/setup.sh                      # weights, onnxruntime + python-dotenv, v2 validation plan list
```

## Run

```bash
# the 4 benchmark plans (90 questions), v1 rules vs. v2 CPU subset (layout + 2 YOLO models)
npm run bench:pipelines -- --questions ../Junction/test_questions.jsonl --images ../Junction

# a sample of the full dataset, without the plans v2 was tuned on
npm run bench:pipelines -- --questions ../Junction/questions.jsonl --images ../Junction \
    --sample 30 --exclude-plans benchmark/v2_validation_plans.txt

# v1 exactly like the app (Gemini agent + vision; costs ≈ $0.02 per question, see README.md 5.1)
npm run bench:pipelines -- ... --v1-agent --v1-vision

# all 7 v2 models: run main.py on the GPU VM (see pipeline_v2/deploy/verda/README.md), then score its facts
npm run bench:pipelines -- ... --v2-facts ~/Downloads/Junction/facts
# one v2 model instead of the merge:
npm run bench:pipelines -- ... --v2-facts <facts> --v2-file yolo_general.campi.json
```

`npm run bench:pipelines -- …` is `python benchmark/compare_pipelines.py …`; `--help` lists all options
(thresholds, OCR model, `--skip-v1`, `--skip-v2`, `--plans`, `--device cuda`, `--v2-python`).

## Output (`benchmark/runs/<timestamp>/`)

| File | Content |
|---|---|
| `report.md` | accuracy per question kind for v1 and v2, how often only one of them was right, time per plan, which v2 models produced output (with a warning if one failed) |
| `results.jsonl` | one line per question: expected, v1 value + verdict + rule explanation, v2 value + verdict + schema field + source model + whether the models disagreed |
| `summary.json` | the same numbers machine-readable, plus all arguments |
| `input/`, `v1/`, `v2/` | staged images, v1 plan graphs, v2 facts, `main.log` and per-model logs |

## Fair comparison – check before drawing conclusions

* **Same thresholds:** v1 defaults to 0.65 for texts, symbols and rooms (as in `docker-compose.yml`); v2 uses its
  per-class thresholds from the model specs.
* **No tuning on the test plans:** v2's model ranking was chosen on its validation split
  (`~/Downloads/Junction/work/val_input`) – exclude those plans (`--exclude-plans`).
* **CPU subset vs. full v2:** the default v2 run uses only `layout`, `yolo_general`, `yolo_symbols` (what runs on a
  laptop); the GPU models (`raster2seq`, `raster2graph`, `plan2strct`, `text_ocr`) only count when their facts come
  from a GPU run.
* **Same layout weights:** both use `model_best_val_loss_var.pkl`; differences in `layout` come from code changes in
  `planqa/` between the branches, not from the model.
* **Rules vs. agent:** compare v1 rules with v2 first (both deterministic, no API costs), then v1 agent separately.
