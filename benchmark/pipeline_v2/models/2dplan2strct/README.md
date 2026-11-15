---
language: en
model_name: 2DPlan2Strct
pipeline_tag: object-detection
tags:
  - floor-plan
  - architecture
  - object-detection
  - rfdetr
metrics:
  - map
license: other
license_name: non-commercial-research-only
license_link: https://huggingface.co/OsamaMo/2dplan2strct/blob/main/USAGE_TERMS.md
---

# 2DPlan2Strct

### From Floor Plan Images to Structured Architectural Understanding

Detect **walls, rooms, doors, and windows** in 2D floor-plan images. Inference runs locally after the weights are downloaded from Hugging Face. It returns a JSON file with detections and an annotated PNG. Walls and rooms are represented by bounding boxes, not segmentation masks.

## Before and after detection

| Before detection | After detection |
| :---: | :---: |
| ![Floor plan before detection](https://huggingface.co/OsamaMo/2dplan2strct/resolve/main/images/before_detection.webp) | ![Same floor plan with detected walls, rooms, doors, and windows highlighted](https://huggingface.co/OsamaMo/2dplan2strct/resolve/main/images/after_detection.png) |
| ![Second floor plan before detection](https://huggingface.co/OsamaMo/2dplan2strct/resolve/main/images/before_rec_2.jpeg) | ![Second floor plan with detected walls, rooms, doors, and windows highlighted](https://huggingface.co/OsamaMo/2dplan2strct/resolve/main/images/after_rec_2.png) |

## Quick start

Use Python 3.10–3.12. In a clean virtual environment, install the requirements and download the inference script:

```bash
python -m pip install "huggingface_hub>=0.25"
hf download OsamaMo/2dplan2strct inference.py requirements.txt --local-dir floorplan-inference
python -m pip install -r floorplan-inference/requirements.txt
python floorplan-inference/inference.py path/to/plan.png --output-dir results/plan
```

The last command downloads and caches the model weights automatically. It writes `results/plan/predictions.json` and `results/plan/annotated.png`. The `--device` option accepts `auto` (default), `cpu`, or `cuda`; `auto` selects CUDA when available. The first run downloads approximately 134 MB of weights.

Raise the confidence threshold to show fewer, stronger predictions:

```bash
python floorplan-inference/inference.py path/to/plan.png --threshold 0.50 --output-dir results/plan
```

The model is hosted on the Hugging Face Hub but is loaded with `rfdetr`. This checkpoint is **not** compatible with Transformers `AutoModel` or `pipeline()`.

## Prediction format

`predictions.json` contains the model ID, input image path, original image width and height, threshold, and a `detections` array. Each detection has:

| Field | Meaning |
| --- | --- |
| `label` | One of `wall`, `room`, `door`, or `window` |
| `score` | Detector confidence from 0 to 1; it is not a calibrated probability |
| `box_xyxy` | `[left, top, right, bottom]` in pixels of the original image |

Detections are sorted from highest to lowest confidence. The annotated PNG keeps the original image size and draws color-coded boxes and labels. The source image is not modified. Overlapping boxes can occur.

## Use in Python

After downloading `inference.py` and installing the requirements, load the model once and reuse it for multiple images:

```python
from inference import FloorPlanDetector, save_prediction

detector = FloorPlanDetector()
result = detector.predict("path/to/plan.png", threshold=0.35)
print(f"Found {len(result['detections'])} objects")
print(result["detections"][:3])
save_prediction(result, "path/to/plan.png", "results/plan")

other_result = detector.predict("path/to/another-plan.jpg", threshold=0.35)
```

Run this snippet from the directory containing the downloaded `inference.py`, or add that directory to your Python path.

## Offline use and reproducible versions

Download the configuration and checkpoint while online, then point the inference script at the local directory:

```bash
hf download OsamaMo/2dplan2strct config.json checkpoint_best_ema.pth --local-dir floorplan-model
python floorplan-inference/inference.py path/to/plan.png --model-dir floorplan-model --output-dir results/plan
```

For a reproducible run, pass a specific Hub commit SHA with `--revision` when using online inference. Use the same revision for any files downloaded separately.

## Evaluation

The published checkpoint is the best EMA checkpoint after epoch 13. The following are **validation** results on 2,964 held-out floor-plan images, as reported by the training run. These are object-detection metrics, not a percentage of floor plans understood correctly or an independent test-set score.

| Metric | Value |
| --- | ---: |
| COCO box mAP@0.5 | 94.52% |
| COCO box mAP@0.5:0.95 | 82.58% |
| F1 | 91.86% |
| Precision | 93.53% |
| Recall | 90.26% |

Scores can change with drawing style, scan quality, and confidence threshold. Inspect detections before using them in a consequential workflow.

## Inference speed

**Measured CPU result for this checkpoint:** On a 1,206 × 1,013 pixel floor plan, the model produced 123 detections at confidence threshold 0.35. On an Intel Xeon Platinum 8488C with two PyTorch CPU threads, median latency was **457.63 ms/image** (about **2.19 images/second**); p95 was **654.66 ms/image**. This used 3 warm-up runs and 20 timed `model.predict()` calls in float32. The RGB image was already loaded; model download, one-time model loading, image-file decoding, and drawing the annotated PNG were outside the timed section.

**GPU reference, not a measurement of this checkpoint:** The [RF-DETR project's published benchmark](https://github.com/roboflow/rf-detr/blob/develop/docs/learn/run/detection.md#pre-trained-checkpoints) reports **4.4 ms/image** for the original RF-DETR Medium at 576 × 576 on an NVIDIA T4 using TensorRT FP16 and batch size 1 (about 227 images/second). This model repository currently provides a PyTorch checkpoint; its actual CUDA latency has not been measured here. The GPU reference uses a different checkpoint and runtime, so it is not a direct CPU-to-GPU speedup comparison.

To measure both devices on your own machine, download `benchmark.py` alongside `inference.py`, then run:

```bash
hf download OsamaMo/2dplan2strct benchmark.py inference.py --local-dir floorplan-inference
python floorplan-inference/benchmark.py path/to/plan.png --device both --warmups 3 --repeats 20
```

The benchmark reports `CUDA: unavailable` when the machine has no CUDA GPU. It synchronizes CUDA before and after each timed call when a GPU is present.

## Model details

- Architecture: RF-DETR Medium, PyTorch checkpoint
- Input: floor-plan image converted to RGB; inference resolution 576 pixels
- Outputs: boxes in `xyxy` image coordinates, class labels, confidence scores
- Classes: `wall`, `room`, `door`, `window`
- Default confidence threshold: `0.35`; tune for your drawings
- Supported use: non-commercial research and evaluation, subject to [usage terms](USAGE_TERMS.md)

## Training setup

The model was trained on an NVIDIA T4 GPU with 16 GB of memory. The reported training duration was 16 hours.

## Download tracking

Hugging Face displays a download count on this model page. The script retrieves `config.json`, which the Hub counts as a model download. The count represents file requests, **not unique people** or model-card readers. To retrieve the current count programmatically:

```python
from huggingface_hub import HfApi

info = HfApi().model_info("OsamaMo/2dplan2strct")
print(info.downloads)
```
