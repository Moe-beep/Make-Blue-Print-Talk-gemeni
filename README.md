# Stoa / PlanQA – explainable answers about construction drawings

Web app for the Junction 2026 challenge *"answer natural-language questions about construction drawings and
building information models accurately and verifiably"*. A user uploads a floor plan (PNG/JPG/PDF) or an IFC model,
asks a question and gets **one clear answer plus the evidence behind it**: which elements were counted, which were
dropped below a confidence threshold, what the LLM agent looked at step by step, and where the vision models are
unsure.

Contents
1. [Quick start](#1-quick-start)
2. [Architecture today](#2-architecture-today)
3. [The multi-model pipeline (`~/Downloads/Junction`)](#3-the-multi-model-pipeline-downloadsjunction)
4. [Integration plan](#4-integration-plan)
5. [Inference costs](#5-inference-costs)
6. [Licences](#6-licences)
7. [Sources](#7-sources)

---

## 1. Quick start

```bash
# keys go into .env (ignored by git), never into .env.example or docker-compose.yml
echo 'GEMINI_API_KEY="..."' >> .env

# layout-model weights (CubiCasa5K, 209 MB, not in git)
mkdir -p pipeline/models && cp <Junction>/models/model_best_val_loss_var.pkl pipeline/models/

docker compose up -d --build        # -> http://localhost:3000
docker compose logs -f
```

`docker-compose.yml` pins the agent model (`GEMINI_MODEL=gemini-flash-latest`, see 5.1 on pinning) and starts the
pipeline with `--vision --min-conf-text 0.65 --min-conf-symbols 0.65 --min-conf-rooms 0.65`.

Development without Docker (two terminals, both need `GEMINI_API_KEY`):

```bash
npm install && npm run dev                     # web app, http://localhost:3000
npm run pipeline                               # python pipeline/serve.py, http://127.0.0.1:8765
npm run test:pipeline                          # offline tests (no models, no network)
```

---

## 2. Architecture today

```
Browser (React, Vite)                       Node / Express (server.ts)                Python (pipeline/serve.py)
┌──────────────────────────┐   /api/planqa/*  ┌─────────────────────┐   HTTP    ┌────────────────────────────────┐
│ 2D viewer (plan + marks) │ ───────────────► │ stream proxy         │ ───────► │ Store: one document per upload │
│ 3D viewer (IFC, meshes)  │                  │ (before JSON parser) │          │  plan: PaddleOCR + CubiCasa5K  │
│ Assistant chat           │   /api/ask ...   │ Gemini direct calls  │          │        → plan graph → explain  │
│  answer card             │ ───────────────► │ (fallback, takeoff,  │          │  IFC:  ifcopenshell → graph    │
│  explanation view        │                  │  voice)              │          │  question: rules + Gemini agent│
└──────────────────────────┘                  └─────────────────────┘          └────────────────────────────────┘
          one container (Dockerfile): Node on :3000 (exposed), Python on 127.0.0.1:8765 (internal)
```

| Part | Where | What it does |
|---|---|---|
| Frontend | `src/` | `ExplanationCard` (answer, confidence, rules vs agent), `ExplanationView` (tabs *Evidence*, *Uncertainty*, *Agent*), `PlanqaPanel` (pipeline status) |
| Proxy | `server.ts` (`/api/planqa/*`) | forwards to `PLANQA_URL` (default `http://127.0.0.1:8765`) |
| Pipeline service | `pipeline/serve.py`, `pipeline/planqa/service_core.py` | upload → background job → `explanation/1`; `POST /documents/<id>/ask` → one explained answer |
| Pipeline core | `pipeline/planqa/` | OCR, layout model, plan graph, `building-graph/1`, rules (`qa.py`), agent (`agent.py`), explanations (`explain.py`) |
| Fallback | `server.ts` (`/api/ask`, `/api/analyze`, …) | direct Gemini answers without verifiable evidence (DXF, meshes, pipeline offline); marked in the UI |

Measured on a laptop CPU (screenshot of a 76 KB JPG): processing **16.2 s**, one agent answer **9.2 s**.

---

## 3. The multi-model pipeline (`~/Downloads/Junction`)

A further-developed branch of the same project. Instead of one layout model it runs **seven vision models in
parallel** on each raster plan, reads **vector PDFs** without any model, converts every model's output to the same
`building-graph/1` schema and fills the question schema `schema_campi.json` field by field.

Entry point: `python main.py <plan or folder> [--device cuda] [--models …] [--remote root@<vm>]`

| Model (`--models`) | Task | Input size | Framework / env | GPU | Weights | Licence of weights |
|---|---|---|---|---|---|---|
| `yolo_general` | 13 classes: symbols, room types, storeys | 1280 px | ONNX (Ultralytics YOLO11m), NMS in graph | no (ONNX) | 81 MB | trained on CubiCasa5K (CC BY-NC-SA 4.0); Ultralytics AGPL-3.0 |
| `yolo_symbols` | window, door, toilet, sink, shower, bathtub, doorway | 1280 px | ONNX (YOLO11l) | no (ONNX) | 102 MB | same as above |
| `layout` | CubiCasa5K segmentation: rooms, icons, storeys | ≤1600 px, 4× TTA | PyTorch (root env) | optional | 208 MB | CC BY-NC 4.0 |
| `plan2strct` | walls, rooms, doors, windows (boxes) | 576 px | RF-DETR medium, own venv | recommended | 133 MB | non-commercial research only |
| `raster2seq` | room / door / window polygons | 256 px (`cubicasa`) or 512 px (`r2g512`) | PyTorch 2.3.1 + cu121, deformable-attention op, own venv | **yes** | 1.13 GB | code MIT, weights CC BY-NC 4.0 |
| `raster2graph` | wall junctions, wall segments, room semantics | – | PyTorch 2.6 + cu126, CUDA-only op, own venv | **yes** | 415 MB | see `Raster-to-Graph/LICENSE` |
| `text_ocr` | every word on the plan (YOLO26-Text + PP-OCRv6) | – | Ultralytics + ONNX Runtime, own venv | recommended | download on first run | Ultralytics AGPL-3.0, PaddleOCR Apache-2.0 |
| `pdf_plan` | room / apartment areas from PDF vector geometry + printed scale | – | pypdfium2, no model | no | – | – |

**How results are combined** (`pipelines/merge_campi.py`): per field a fixed model ranking decides (first model
with a value wins); a scalar on which a lower-ranked model disagrees goes to `_meta.uncertain_fields`;
`_meta.values_by_model` keeps every model's value. The ranking was chosen on a separate validation split of 30
CubiCasa5K plans (not the 30 test plans), e.g. exact door counts: 2DPlan2Strct 15/30, layout 11/30, Raster2Seq 6/30;
toilets / sinks / storeys: layout 24, 22, 30 of 30. Output per plan: `<model>.building_graph.json`,
`<model>.campi.json`, `merged.campi.json`, `house.json`.

**Runtime environment:** each repository needs its own PyTorch/CUDA build, so `pipelines/runners.py` starts every
model as a separate process with its own `.venv`. `deploy/verda/setup.sh` builds all environments on a GPU VM
(Ubuntu 24.04 + CUDA 12, nvcc 12 required for Raster2Seq). `pipelines/remote.py` can run `main.py` on that VM over
SSH and copy the facts back.

**Measured GPU run** (`work/vm_facts`, 30 plans in one batch, per-model process time incl. model loading):
Raster2Seq 39.4 s, Raster-to-Graph 29.0 s, layout 21.9 s, 2DPlan2Strct 14.5 s – run in parallel, so roughly
**40 s wall time for 30 plans (≈ 1.3 s per plan)**. Times of `text_ocr` and the two YOLO models were not in that
run's record.

**Benchmark against the app pipeline:** the branch is copied to `benchmark/pipeline_v2/` and can be compared with
the app pipeline on the same questions without touching the app – see [`benchmark/README.md`](benchmark/README.md)
(`npm run bench:setup`, then `npm run bench:pipelines -- --questions … --images …`).

### What this branch has that the app does not (and vice versa)

| | `~/Downloads/Junction` | `Make_Blueprint_Talk/pipeline` |
|---|---|---|
| Vision models | 7 + vector-PDF reader, merged with model ranking | PaddleOCR + CubiCasa5K layout |
| `planqa/` | newer `build_graph`, `layout`, `plan_to_graph`, `campi_map`, `building_schema` | older versions of those files |
| Explanations | – | `explain.py`, `explanation_schema.py`, `service_core.py`, `serve.py`, English UI texts |
| Question answering | fills `schema_campi.json` (deterministic facts) | rules + Gemini agent with tool trace |

---

## 4. Integration plan

Goal: keep the app's explainable answers and add the stronger detections and the **cross-model agreement** as a
new, real uncertainty signal – without making the web container need a GPU.

### Target architecture

```
                         ┌──────────────────────────────── CPU host (web container) ──────────────────────────┐
Browser ── Node :3000 ──►│ pipeline/serve.py                                                                     │
                         │  • IFC → graph, agent, explanations                                                   │
                         │  • vector PDF → pdf_plan (no model)                                                   │
                         │  • raster plan, CPU tier: YOLO ONNX ×2 + CubiCasa layout + PaddleOCR                   │
                         │  • merge_campi → merged facts + uncertain_fields → agent + explanation               │
                         └───────────────▲──────────────────────────────────────────────────────────────────────┘
                                         │ optional, PLANQA_GPU_URL
                         ┌───────────────┴─────────── GPU worker (Verda RTX 6000 Ada / L40S) ────────────────────┐
                         │ main.py models in their own venvs: raster2seq, raster2graph, plan2strct, text_ocr     │
                         │ small HTTP job API: POST image → per-model building_graph + campi                     │
                         └───────────────────────────────────────────────────────────────────────────────────────┘
```

### Steps

1. **Merge the two `planqa/` branches.** Take the newer extraction files from `~/Downloads/Junction/planqa`
   (`build_graph`, `layout`, `plan_to_graph`, `campi_map`, `building_schema`, `ocr`, `pipeline`, `agent`), keep
   `explain.py`, `explanation_schema.py`, `service_core.py` from the app. Re-run all offline tests of both branches
   (`tests/test_*_offline.py`); fix the explanation builder where the plan-graph format changed.
2. **Copy `pipelines/` into `pipeline/pipelines/`** (`to_graph`, `campi_fill`, `merge_campi`, `pdf_plan`,
   `yolo_infer`, `house`, `schema_fill`) plus `schema_campi.json` and the YOLO ONNX + JSON specs.
3. **CPU tier in `service_core.Store._process_plan`:** run `yolo_general` and `yolo_symbols` with ONNX Runtime
   (no PyTorch, no GPU) next to the existing OCR + layout model, convert each with `pipelines/to_graph.py`, merge with
   `merge_campi`. PDFs: try `pdf_plan.extract_pdf_plan` first (exact areas from the vector geometry) and fall back to
   rendering + raster models.
4. **GPU tier (optional):** wrap `main.py --local` in a small HTTP job server on the GPU VM (same pattern as
   `serve.py`: upload, background job, poll). `serve.py` calls it when `PLANQA_GPU_URL` is set and merges the extra
   models in; without it the CPU tier answers alone. Build the worker image from `deploy/verda/setup.sh`
   (CUDA 12 base image, one venv per model).
5. **Agent on merged data:** give the agent the merged `building-graph` (chosen model per field) as today, plus one
   new tool `model_values(field)` that returns `values_by_model` and whether the field is in `uncertain_fields`. The
   rules estimate (`qa.py`) stays the reference the agent has to justify deviations from.
6. **Explanations (UI):** new card *Model agreement* in the *Evidence* tab: per answer field, the value of every
   model, which model the ranking picked and why (validation score), and a warning when models disagree. This is the
   cross-model distribution the mock-ups showed – now from real data instead of a single model's entropy. Evidence
   boxes come from the model that supplied the value (`_meta.sources`).
7. **Docker:** web image + `onnxruntime` and the two ONNX files (≈ 180 MB). GPU worker: separate image, separate
   compose service / VM; never in the web image.
8. **Measure before switching defaults:** run `evaluate_campi.py` and the app's benchmark (`answer.py`) on the
   held-out split with CPU tier only, GPU tier, and both; pick the default from those numbers (no tuning on the 4
   benchmark plans).

### Risks

* Seven models multiply licence obligations (section 6) – the deciding factor for any public deployment.
* `raster2seq` and `raster2graph` need CUDA ops: no CPU fallback, so the GPU tier must stay optional.
* The `gemini-flash-latest` alias is repointed with every Flash release; pin a version for reproducible results.

---

## 5. Inference costs

All prices in USD, excluding VAT, checked 15 Nov 2026 – they change often, verify before budgeting.

### 5.1 LLM agent (Gemini)

The agent uses `gemini-flash-latest`. This alias points to the newest Flash model and is switched with every Flash
release; since 2 Sep 2026 that is **Gemini 3.8 Flash** (`gemini-3.8-flash`), so the benchmark runs of 14 Nov 2026
very likely ran on it. Recommendation: set `GEMINI_MODEL=gemini-3.8-flash` to pin it.

| Model | Input / 1M tokens | Output / 1M tokens (incl. thinking) | Note |
|---|---|---|---|
| Gemini 3.8 Flash | $0.75 | $3.75 | introductory, until 31 Dec 2026 |
| Gemini 3.8 Flash | $1.50 | $7.50 | from 1 Jan 2027 |
| Gemini 3.8 Flash, Batch | $0.375 | $1.875 | introductory; not usable for interactive chat |
| Gemini 3.5 Flash-Lite | $0.30 | $2.50 | cheaper alternative, not evaluated |
| Gemini 3.1 Pro | $2.00 | $12.00 | ≤ 200k context |

**Measured token use per question** (`out/predictions.jsonl`, 90 plan questions with `--vision`;
`out/agent_predictions.jsonl`, 96 IFC questions; thinking level *low*; 4.5 agent steps on average):

| Question type | Input tokens (mean / median / p90 / max) | Output tokens (mean / p90) | Time per answer |
|---|---|---|---|
| Floor plan (image) | 24,590 / 19,238 / 44,913 / 72,527 | 334 / 555 | 15.2 s mean |
| IFC model | 28,081 / 16,963 / 60,752 / 139,471 | 430 / 787 | 12.0 s mean |

Input dominates because the whole conversation (overview, tool results, image crops) is re-sent at every agent
step. No thinking tokens were reported in these runs; with a higher `--thinking` level they are billed as output.

**Cost per question:**

| Price | Floor plan, mean | Floor plan, p90 | IFC, mean | IFC, p90 | per 1,000 questions (plan / IFC) |
|---|---|---|---|---|---|
| 3.8 Flash, 2026 intro | $0.020 | $0.036 | $0.023 | $0.049 | $19.69 / $22.67 |
| 3.8 Flash, from 2027 | $0.039 | $0.072 | $0.045 | $0.097 | $39.39 / $45.34 |
| 3.5 Flash-Lite | $0.008 | $0.015 | $0.010 | $0.020 | $8.21 / $9.50 |
| 3.1 Pro | $0.053 | $0.097 | $0.061 | $0.131 | $53.18 / $61.32 |

Not included: the LLM judge used only in benchmarks (`--judge`), and the app's direct Gemini paths (fallback
answers, *Analyze Space & Objects*, 2D locating, voice). Those use model ids written by AI Studio
(`gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-robotics-er-*`, `gemini-3.5-transcribe`,
`gemini-3.8-flash-lite-tts`); I could not find official prices for all of them – third parties list Gemini 4 Carbon at
$2 / $10 per 1M tokens. Pin these to priced models before going public. ElevenLabs voice (optional) is billed by
ElevenLabs' own plans.

### 5.2 Vision models

The vision models run on our own hardware: their cost is server time, not per call.

| Option | Price | What it runs | Per plan |
|---|---|---|---|
| CPU host (today), e.g. Hetzner CCX33, 8 dedicated vCPU, 32 GB | €138.49 / month (price since 15 Jun 2026) | web app, OCR + CubiCasa layout, IFC, agent calls; CPU tier incl. 2 YOLO ONNX | ≈ 16 s CPU; marginal cost ≈ 0 at low load |
| Verda 1× RTX 6000 Ada (48 GB), on demand | $0.83 / h → ≈ $606 / month 24/7 | all 7 models (recommended GPU in `deploy/verda/README.md`) | ≈ 1.3 s → ≈ $0.0003 when busy |
| Verda 1× RTX 6000 Ada, spot | $0.29 / h → ≈ $212 / month 24/7 | same, can be interrupted | same |
| Verda 1× L40S (48 GB) | ≈ $0.91 / h (third-party listing) | same | same |

The GPU itself is cheap per plan; **the bill is idle time**. A stopped Verda instance still bills its volume
(`deploy/verda/README.md`). For a demo, starting the VM for a few hours ($0.83 × 4 h ≈ $3.30) is enough; a 24/7 GPU
only pays off with steady traffic.

### 5.3 Scenarios (per month)

| Scenario | Plans | Questions | LLM (2026 / 2027 prices) | Hosting | Total (approx.) |
|---|---|---|---|---|---|
| Hackathon demo | 50 | 250 | $5 / $10 | own laptop (≈ $0) or CPU host €138; GPU on demand 4 h ≈ $3 | **≈ $8–165** |
| Pilot, CPU tier only | 1,000 | 5,000 | $100 / $200 | CPU host €138 | **≈ $250–350** |
| Pilot with GPU tier, spot | 1,000 | 5,000 | $100 / $200 | CPU host €138 + spot GPU $212 | **≈ $460–560** |
| Pilot with GPU tier, on demand 24/7 | 1,000 | 5,000 | $100 / $200 | CPU host €138 + GPU $606 | **≈ $855–955** |

LLM column uses a mean of $0.02 (2026) / $0.04 (2027) per question; €138 ≈ $150.

### 5.4 Cost levers

* **Pin the model** and re-measure tokens per question after every model change (newer Flash models used more
  tokens per task in public benchmarks).
* **Fewer re-sent tokens:** the agent re-sends the full history at each step; context caching of the document
  overview (cached input on 3.8 Flash: $0.075 / 1M) or a smaller overview cuts the input side, which is ~95 % of the
  LLM cost.
* `--vision` adds image crops (up to 3 per question) to the input; turn it off where labels are reliable.
* Answer fixed question templates with the deterministic facts (`merged.campi.json`) and call the agent only for
  free-text questions (open point 7 in `PROJECT_NOTES.md`).
* Batch API (50 % off) for offline evaluation runs, never for the chat.
* GPU: spot instances, scale to zero, or CPU tier only (YOLO ONNX already gives the best door / window counts in
  the validation ranking).

---

## 6. Licences

| Component | Licence | Consequence |
|---|---|---|
| CubiCasa5K weights + `cubicasa_net.py` | CC BY-NC 4.0 | non-commercial only |
| YOLO weights (trained on CubiCasa5K) | CC BY-NC-SA 4.0 data; Ultralytics AGPL-3.0 | non-commercial; AGPL also covers network use: serving it publicly requires publishing the source or an Ultralytics enterprise licence |
| 2DPlan2Strct weights | non-commercial research and evaluation only | no commercial use, no redistribution without these terms |
| Raster2Seq | code MIT, weights CC BY-NC 4.0 | non-commercial only |
| Raster-to-Graph | see `Raster-to-Graph/LICENSE` | check before deployment |
| PaddleOCR | Apache-2.0 | fine |
| Gemini API | Google API terms; data-use rules differ between free and paid tier | check the terms before sending real customer drawings |

Fine for the hackathon and research; a commercial or public service needs other weights or licences.

---

## 7. Sources

* Measurements: `Junction/out/predictions.jsonl`, `Junction/out/agent_predictions.jsonl` (token counts, steps,
  seconds), `~/Downloads/Junction/work/vm_facts/*/merged.campi.json` (GPU run times),
  `~/Downloads/Junction/pipelines/merge_campi.py` (validation ranking), `~/Downloads/Junction/deploy/verda/README.md`.
* Gemini 3.8 Flash prices, introductory period and 2027 prices:
  [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing),
  [Gemini 3.8 Flash guide](https://developer.tenten.co/gemini-3-8-flash),
  [benchlm.ai Gemini pricing table](https://benchlm.ai/google/api-pricing)
* `gemini-flash-latest` alias behaviour: [Gemini API models](https://ai.google.dev/gemini-api/docs/models),
  [Gemini API release notes](https://ai.google.dev/gemini-api/docs/changelog),
  [Gemini 3.8 Flash launch](https://howaiworks.ai/google-gemini-3-8-flash-launch-2026/)
* Verda GPU prices: [verda.com/products](https://verda.com/products); L40S:
  [DeployBase](https://deploybase.ai/articles/verda-gpu-cloud-pricing-complete-guide-hr-for-every-gpu)
* Hetzner CCX33 after the June 2026 price change:
  [Northflank](https://northflank.com/blog/hetzner-cloud-server-price-increases)
* Gemini 4 Carbon third-party price: [computeprices.com](https://computeprices.com/providers/gmi-cloud/models/gemini-4-carbon)
