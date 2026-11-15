# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .raster2seq import build


def build_model(args, tokenizer=None):
    if not args.poly2seq:
        raise ValueError("Only the Raster2Seq model (--poly2seq) is supported in this inference-only build.")
    return build(args, tokenizer=tokenizer)
