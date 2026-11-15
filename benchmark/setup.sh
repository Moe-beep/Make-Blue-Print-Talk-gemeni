#!/usr/bin/env bash
# Prepare the pipeline benchmark: model weights for both pipelines, the extra Python packages, and the list of
# plans pipeline_v2 was tuned on (to exclude them). Run from the project root, in the Python env of the pipeline:
#     bash benchmark/setup.sh
# Sources (override with env vars):
#     V1_SRC  folder with models/model_best_val_loss_var.pkl   (default ~/PycharmProjects/Junction)
#     V2_SRC  the multi-model project with models/ and work/   (default ~/Downloads/Junction)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V1_SRC="${V1_SRC:-$HOME/PycharmProjects/Junction}"
V2_SRC="${V2_SRC:-$HOME/Downloads/Junction}"
V2="$ROOT/benchmark/pipeline_v2"
CUBICASA=model_best_val_loss_var.pkl

say() { printf '  %-58s %s\n' "$1" "$2"; }

echo "== layout weights (CubiCasa5K)"
mkdir -p "$ROOT/pipeline/models" "$V2/models"
if [[ ! -f "$ROOT/pipeline/models/$CUBICASA" ]]; then
  src=""
  for c in "$V1_SRC/models/$CUBICASA" "$V2_SRC/models/$CUBICASA"; do [[ -f "$c" ]] && src="$c" && break; done
  if [[ -n "$src" ]]; then
    cp "$src" "$ROOT/pipeline/models/$CUBICASA"   # a real copy: the app's Docker build needs the file itself
    say "pipeline/models/$CUBICASA" "copied from $src"
  else
    say "pipeline/models/$CUBICASA" "MISSING (not found in $V1_SRC or $V2_SRC)"
  fi
else
  say "pipeline/models/$CUBICASA" "ok"
fi
if [[ ! -e "$V2/models/$CUBICASA" && -f "$ROOT/pipeline/models/$CUBICASA" ]]; then
  ln -s "$ROOT/pipeline/models/$CUBICASA" "$V2/models/$CUBICASA"
fi
say "benchmark/pipeline_v2/models/$CUBICASA" "$([[ -e "$V2/models/$CUBICASA" ]] && echo ok || echo MISSING)"

echo "== pipeline_v2 weights (links into $V2_SRC/models, not copied)"
link() {   # link <relative path under models/>
  local rel="$1"
  if [[ -e "$V2/models/$rel" ]]; then say "models/$rel" "ok"; return; fi
  if [[ -e "$V2_SRC/models/$rel" ]]; then
    mkdir -p "$(dirname "$V2/models/$rel")"
    ln -s "$V2_SRC/models/$rel" "$V2/models/$rel"
    say "models/$rel" "linked"
  else
    say "models/$rel" "missing (model will fail; only needed if you run it)"
  fi
}
link yolo-weights/floorplan-yolo11m-1280.onnx            # yolo_general  (CPU ok)
link yolo-weights/floorplan-symbols-yolo11l-1280.onnx    # yolo_symbols  (CPU ok)
link 2dplan2strct/checkpoint_best_ema.pth                # plan2strct    (own venv)
link graph_topology/graph.pth                            # raster2graph  (CUDA only)
link Multi_task_raster/Raster2Seq.pth                    # raster2seq r2g512 profile (CUDA)

echo "== Python packages for the CPU subset (layout, yolo_general, yolo_symbols)"
python -m pip install --quiet onnxruntime python-dotenv && say "onnxruntime, python-dotenv" "installed"

echo "== plans pipeline_v2 was tuned on (its validation split) -> excluded from comparisons"
VAL="$ROOT/benchmark/v2_validation_plans.txt"
if [[ -d "$V2_SRC/work/val_input" ]]; then
  find "$V2_SRC/work/val_input" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    -printf '%f\n' | sed -E 's/\.[^.]+$//; s/[^A-Za-z0-9_-]/_/g' | sort > "$VAL"
  say "benchmark/v2_validation_plans.txt" "$(wc -l < "$VAL") plans"
else
  say "benchmark/v2_validation_plans.txt" "not created ($V2_SRC/work/val_input not found)"
fi
echo "done. Next: see benchmark/README.md"
