import numpy as np
from PIL import Image


def resize_and_pad(image: np.ndarray, target_size: tuple, pad_value: int = 0, interp=Image.BICUBIC) -> np.ndarray:
    """Resize a uint8 HxW or HxWxC image preserving aspect ratio, then center-pad it to `target_size` (h, w)."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    is_single_channel = image.ndim == 3 and image.shape[2] == 1
    pil_image = Image.fromarray(image[:, :, 0] if is_single_channel else image)
    resized = np.asarray(pil_image.resize((new_w, new_h), interp))
    if is_single_channel:
        resized = resized[:, :, None]

    pad_h, pad_w = target_size[0] - new_h, target_size[1] - new_w
    top, left = pad_h // 2, pad_w // 2
    pad_width = [(top, pad_h - top), (left, pad_w - left)] + [(0, 0)] * (resized.ndim - 2)
    return np.pad(resized, pad_width, mode="constant", constant_values=pad_value)
