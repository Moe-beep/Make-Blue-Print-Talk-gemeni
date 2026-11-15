# pipeline/ – PlanQA (Python)

Copy of the Junction project (`planqa/`, `answer.py`, `serve.py`) that the web app needs at runtime: OCR + CubiCasa
layout model → plan graph, IFC → building graph, Gemini agent with tool trace, explanations (`explanation/1`).
Background and design decisions: `PROJECT_NOTES.md` in the Junction project.

```
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
cp <Junction>/models/model_best_val_loss_var.pkl models/      # layout weights, 209 MB, CC BY-NC 4.0, not in git
export GEMINI_API_KEY=...
python serve.py [--vision]                                      # http://127.0.0.1:8765, used via /api/planqa
python tests/test_service_offline.py                            # offline tests (no models, no network)
```

`data/high_quality_architectural-10498.png` is only the fixture of the offline tests. In Docker (`../Dockerfile`) this
service runs next to the Node server; only the Node port is exposed.
