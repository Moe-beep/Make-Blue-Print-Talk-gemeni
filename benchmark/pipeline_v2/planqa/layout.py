"""Room layout with the pretrained CubiCasa5K model (CPU).

Weights: models/model_best_val_loss_var.pkl (official checkpoint, link in the
CubiCasa5k README). Needs: pip install torch --index-url https://download.pytorch.org/whl/cpu

segment()                -> class probabilities (torch)
rooms_from_segmentation() -> room polygons + fixtures (numpy/OpenCV, testable without torch)
walls_from_segmentation() -> straight wall / railing runs of the Wall and Railing classes
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import Fixture, GraphObject, GraphRoom, GraphWall, RoomType

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath",
                "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
ICON_CLASSES = ["Empty", "Window", "Door", "Closet", "Electrical appliance", "Toilet", "Sink",
                "Sauna bench", "Fire place", "Bathtub", "Chimney"]
ROOM_TO_TYPE = {1: RoomType.balcony, 3: RoomType.kitchen, 4: RoomType.living_room, 5: RoomType.bedroom,
                6: RoomType.bathroom, 7: RoomType.hallway, 9: RoomType.storage, 10: RoomType.garage,
                11: RoomType.other}
ICON_TO_FIXTURE = {3: Fixture.wardrobe, 5: Fixture.toilet, 6: Fixture.sink, 7: Fixture.sauna_bench,
                   8: Fixture.fireplace, 9: Fixture.bathtub}
ICON_TO_OBJECT = {1: "window", 2: "door", 3: "closet", 4: "electrical_appliance", 5: "toilet", 6: "sink",
                  7: "sauna_bench", 8: "fireplace", 9: "bathtub", 10: "chimney"}
WALL_TO_KIND = {2: "wall", 8: "railing"}
N_HEATMAPS = 21   # junction heatmaps (wall corners, opening ends, icon corners) before the room channels
DEFAULT_WEIGHTS = os.environ.get("CUBICASA_WEIGHTS", "models/model_best_val_loss_var.pkl")


class LayoutModel:
    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str = "cpu", threads: int | None = None):
        import torch
        from .cubicasa_net import FloorplanNet

        if threads:
            torch.set_num_threads(threads)
        if not Path(weights).is_file():
            raise FileNotFoundError(f"weights not found: {weights} (model_best_val_loss_var.pkl, see CubiCasa5k README)")
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # official, trusted checkpoint
        self.model = FloorplanNet(44)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()
        self.torch, self.device = torch, device

    def segment(self, img: Image.Image, max_side: int = 1600, tta: bool = True):
        """(room_prob [12,H,W], icon_prob [11,H,W]) at original size.
        Downscale to max_side, pad to a multiple of 64, normalize to [-1, 1] as in training.
        tta: average over 4 rotations (4x slower, more stable).
        The 21 junction heatmaps (sigmoid, working resolution) are kept in self.last_heatmaps."""
        torch, F = self.torch, self.torch.nn.functional
        W, H = img.size
        s = min(1.0, max_side / max(W, H))
        work = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
        arr = np.asarray(work.convert("RGB"), dtype=np.float32) / 127.5 - 1
        h, w = arr.shape[:2]
        arr = np.pad(arr, ((0, (-h) % 64), (0, (-w) % 64), (0, 0)), constant_values=1.0)  # white padding
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device)

        rooms = icons = heat = 0
        turns = (0, 1, 2, 3) if tta else (0,)
        with torch.no_grad():
            for k in turns:
                out = torch.rot90(self.model(torch.rot90(x, k, (2, 3))), -k, (2, 3))
                rooms = rooms + F.softmax(out[:, 21:33], 1)
                icons = icons + F.softmax(out[:, 33:44], 1)
                heat = heat + torch.sigmoid(out[:, :N_HEATMAPS])
        self.last_heatmaps = (heat / len(turns))[0, :, :h, :w].cpu().numpy()
        rooms = (rooms / len(turns))[0, :, :h, :w].cpu().numpy()
        icons = (icons / len(turns))[0, :, :h, :w].cpu().numpy()
        if s < 1:
            rooms, icons = (np.stack([cv2.resize(c, (W, H)) for c in p]) for p in (rooms, icons))
        return rooms, icons


def rooms_from_segmentation(room_prob: np.ndarray, icon_prob: np.ndarray,
                            min_area_frac: float = 0.002, min_icon_px: int = 12) -> list[GraphRoom]:
    """One polygon per connected region of each room class, with confidence and fixtures inside."""
    _, H, W = room_prob.shape
    room_map, room_conf, icon_map = room_prob.argmax(0), room_prob.max(0), icon_prob.argmax(0)
    min_area = max(30, int(min_area_frac * H * W))
    kernel = np.ones((3, 3), np.uint8)

    rooms = []
    for cls, rtype in ROOM_TO_TYPE.items():
        mask = cv2.morphologyEx((room_map == cls).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            comp = (labels == i).astype(np.uint8)
            cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            poly = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True)[:, 0, :]
            if len(poly) < 3:
                continue
            conf = float(room_conf[comp > 0].mean())
            counts = np.bincount(icon_map[cv2.dilate(comp, kernel, iterations=2) > 0], minlength=len(ICON_CLASSES))
            fixtures = [f for k, f in ICON_TO_FIXTURE.items() if counts[k] >= min_icon_px]

            # CubiCasa has no sauna class: a bath in which sauna benches dominate is a sauna
            if rtype == RoomType.bathroom and counts[7] >= min_icon_px and counts[7] > max(counts[5], counts[6], counts[9]):
                rtype_final = RoomType.sauna
            else:
                rtype_final = rtype
            reason = f"CubiCasa5K class '{ROOM_CLASSES[cls]}' (mean prob {conf:.2f})"
            if fixtures:
                reason += "; icons: " + ", ".join(f.value for f in fixtures)
            x, y, w, h = cv2.boundingRect(poly)
            rooms.append(GraphRoom(
                id="", room_type=rtype_final, model_class=ROOM_CLASSES[cls], type_conf=round(conf, 3),
                type_reason=reason, fixtures=fixtures, polygon_px=poly.tolist(),
                box_px=[x, y, x + w, y + h], area_px=float(comp.sum())))

    rooms.sort(key=lambda r: (r.box_px[1], r.box_px[0]))
    for i, r in enumerate(rooms, 1):
        r.id = f"r{i:02d}"
    return rooms


def objects_from_segmentation(icon_prob: np.ndarray, min_area_frac: float = 0.00004) -> list[GraphObject]:
    """Symbol instances = connected regions per icon class (doors, windows, toilets, sinks, ...).
    Touching symbols of the same class merge, which is what we want for a double-bowl sink."""
    _, H, W = icon_prob.shape
    icon_map, icon_conf = icon_prob.argmax(0), icon_prob.max(0)
    min_area = max(8, int(min_area_frac * H * W))
    objs = []
    for cls, kind in ICON_TO_OBJECT.items():
        mask = cv2.morphologyEx((icon_map == cls).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            objs.append(GraphObject(id="", kind=kind, box_px=[int(x), int(y), int(x + w), int(y + h)], area_px=int(area),
                                    conf=round(float(icon_conf[labels == i].mean()), 3)))
    objs.sort(key=lambda o: (o.kind, o.box_px[1], o.box_px[0]))
    for i, o in enumerate(objs, 1):
        o.id = f"o{i:03d}"
    return objs


def floors_from_segmentation(room_prob: np.ndarray, min_rel_size: float = 0.35) -> list[list[int]]:
    """Separate storey drawings on one sheet = separate building outlines.
    Everything that is not background is dilated a little (bridges door gaps and
    thin balconies) and split into connected parts; tiny parts (detached sheds,
    legend boxes) are ignored relative to the largest outline."""
    _, H, W = room_prob.shape
    fg = (room_prob.argmax(0) != 0).astype(np.uint8)
    k = max(3, int(0.025 * max(H, W)))  # separate storey drawings are far apart; corridors may be missed
    fg = cv2.dilate(fg, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return []
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, a in stats[1:] if a >= min_rel_size * largest]


def walls_from_segmentation(room_prob: np.ndarray, min_len_frac: float = 0.01) -> list[GraphWall]:
    """Straight runs of the Wall / Railing classes. Plans are mostly orthogonal, so a morphological
    opening with a long horizontal (vertical) line keeps the horizontal (vertical) runs; each connected
    run is one piece. Diagonal or curved walls are not returned."""
    _, H, W = room_prob.shape
    room_map, room_conf = room_prob.argmax(0), room_prob.max(0)
    walls = []
    for cls, kind in WALL_TO_KIND.items():
        mask = (room_map == cls).astype(np.uint8)
        if not mask.any():
            continue
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        typical = max(2.0, 2 * float(mask.sum()) / max(perimeter, 1.0))   # thin strip: area = L*t, perimeter ~ 2L
        run = max(int(3 * typical), int(min_len_frac * max(H, W)), 5)
        for orient, kernel in (("h", np.ones((1, run), np.uint8)), ("v", np.ones((run, 1), np.uint8))):
            runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=4)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                length = w if orient == "h" else h
                walls.append(GraphWall(id="", kind=kind, box_px=[x, y, x + w, y + h], length_px=float(length),
                                       thickness_px=round(area / length, 1), orientation=orient,
                                       conf=round(float(room_conf[labels == i].mean()), 3)))
    walls.sort(key=lambda w: (w.kind, w.box_px[1], w.box_px[0]))
    for i, w in enumerate(walls, 1):
        w.id = f"w{i:03d}"
    return walls
