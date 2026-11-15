"""Raster2Seq checkpoint profiles: each checkpoint needs its own predict.py flags and has its own label set.

r2g512   = models/Multi_task_raster/Raster2Seq.pth (saved args: dataset r2g, 13 classes, 512 px).
           Rooms only: the Raster2Graph dataset has no door/window classes.
cubicasa = hf:cubicasa5k (downloaded on first use). Rooms + doors + windows, 256 px.
Flags copied from Raster2Seq/tools/predict_r2g.sh and predict_cc5k.sh.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON_ARGS = (
    "--input_channels=3",
    "--poly2seq",
    "--seq_len=512",
    "--num_bins=32",
    "--disable_poly_refine",
    "--dec_attn_concat_src",
    "--per_token_sem_loss",
    "--use_anchor",
    "--ema4eval",
    "--save_pred",
    "--num_workers=0",
)

# Raster2Seq/util/plot_utils.py R2G_LABEL / CC5K_LABEL
R2G_LABELS = {
    0: "unknown",
    1: "living_room",
    2: "kitchen",
    3: "bedroom",
    4: "bathroom",
    5: "restroom",
    6: "balcony",
    7: "closet",
    8: "corridor",
    9: "washing_room",
    10: "PS",
    11: "outside",
}
CC5K_LABELS = {
    0: "Outdoor",
    1: "Kitchen",
    2: "Living Room",
    3: "Bed Room",
    4: "Bath",
    5: "Entry",
    6: "Storage",
    7: "Garage",
    8: "Undefined",
    9: "Window",
    10: "Door",
}
DOOR_LABEL = "Door"
WINDOW_LABEL = "Window"


@dataclass(frozen=True)
class Raster2SeqProfile:
    checkpoint: str  # local path or hf:<key>
    args: tuple[str, ...]
    labels: dict[int, str]
    canvas_size: int
    has_openings: bool  # True when the label set includes doors and windows


PROFILES = {
    "r2g512": Raster2SeqProfile(
        checkpoint=str(ROOT / "models" / "Multi_task_raster" / "Raster2Seq.pth"),
        args=(
            "--dataset_name=r2g",
            "--semantic_classes=13",
            "--image_size=512",
            *COMMON_ARGS,
        ),
        labels=R2G_LABELS,
        canvas_size=512,
        has_openings=False,
    ),
    "cubicasa": Raster2SeqProfile(
        checkpoint="hf:cubicasa5k",
        args=(
            "--dataset_name=cubicasa",
            "--semantic_classes=12",
            "--image_size=256",
            *COMMON_ARGS,
        ),
        labels=CC5K_LABELS,
        canvas_size=256,
        has_openings=True,
    ),
}
# cubicasa: the question images are CubiCasa5K plans and doors/windows need its classes
DEFAULT_PROFILE = "cubicasa"
