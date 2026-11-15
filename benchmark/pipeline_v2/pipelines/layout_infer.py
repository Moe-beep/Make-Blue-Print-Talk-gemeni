"""CubiCasa5K layout model (planqa/layout.py) as a standalone job: symbols, rooms and storey outlines
per image -> <output>/<stem>.json.

Run from the project root (main.py does this in parallel with the other models):
    python -m pipelines.layout_infer work/input --output work/layout --weights models/model_best_val_loss_var.pkl

All detections are written with their confidence; the confidence threshold is applied when merging
(pipelines/parsers.py), so it can be changed without rerunning the model.
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from planqa.layout import (
    LayoutModel,
    floors_from_segmentation,
    objects_from_segmentation,
    rooms_from_segmentation,
)

logger = logging.getLogger("layout")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_SIDE = 1600  # same as planqa/pipeline.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CubiCasa5K layout model: doors, windows and other symbols"
    )
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, required=True, help="model_best_val_loss_var.pkl"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="skip the 4-rotation averaging (4x faster)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    images = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {args.input}")

    model = LayoutModel(str(args.weights), device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        room_prob, icon_prob = model.segment(image, MAX_SIDE, tta=not args.no_tta)
        objects = objects_from_segmentation(icon_prob)
        rooms = rooms_from_segmentation(room_prob, icon_prob)
        result = {
            "image": image_path.name,
            "image_size_px": list(image.size),
            "objects": [
                {
                    "kind": o.kind,
                    "conf": o.conf,
                    "box_px": o.box_px,
                    "area_px": o.area_px,
                }
                for o in objects
            ],
            "rooms": [
                {
                    "room_type": r.room_type.value,
                    "model_class": r.model_class,
                    "type_conf": r.type_conf,
                    "fixtures": [f.value for f in r.fixtures],
                    "polygon_px": r.polygon_px,
                    "box_px": r.box_px,
                    "area_px": r.area_px,
                }
                for r in rooms
            ],
            # separate storey drawings on the sheet (outline boxes)
            "floors": floors_from_segmentation(room_prob),
        }
        out_path = args.output / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(result), encoding="utf-8")
        logger.info("%s -> %s (%d symbols, %d rooms)", image_path.name, out_path, len(objects), len(rooms))


if __name__ == "__main__":
    main()
