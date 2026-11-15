"""Data model of the plan graph: rooms (layout model) + texts (OCR) + links."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class RoomType(str, Enum):
    bathroom = "bathroom"
    sauna = "sauna"
    kitchen = "kitchen"
    living_room = "living_room"
    bedroom = "bedroom"
    hallway = "hallway"
    storage = "storage"
    garage = "garage"
    balcony = "balcony"   # CubiCasa "Outdoor"
    other = "other"


class Fixture(str, Enum):
    toilet = "toilet"
    sink = "sink"
    bathtub = "bathtub"
    sauna_bench = "sauna_bench"
    fireplace = "fireplace"
    wardrobe = "wardrobe"


class TextKind(str, Enum):
    room_label = "room_label"   # short word: MH, KPH, WC, Bad
    area = "area"               # '5,8 m2', '120 sq ft'
    dimension = "dimension"     # plain number: '3600', '4,20'
    other = "other"


class GraphText(BaseModel):
    id: str
    text: str
    kind: TextKind = TextKind.other
    box_px: list[int]                  # [x0, y0, x1, y1] original pixels
    conf: Optional[float] = None       # OCR confidence
    flags: list[str] = []              # low_confidence, number_without_unit
    room_id: Optional[str] = None
    value: Optional[float] = None      # area in m² or dimension value
    unit: Optional[str] = None         # m2 | sqft


class GraphRoom(BaseModel):
    id: str
    room_type: RoomType
    model_class: str                   # original CubiCasa class, e.g. "Bath"
    type_conf: float                   # mean class probability over the room
    type_reason: str                   # human-readable evidence
    fixtures: list[Fixture] = []
    polygon_px: list[list[int]]        # outline [[x, y], ...]
    box_px: list[int]
    area_px: float
    label_text: Optional[str] = None   # OCR label inside the room
    label_text_ids: list[str] = []
    area_m2: Optional[float] = None    # OCR area label inside the room
    area_text_ids: list[str] = []
    issues: list[str] = []             # no_area_label, multiple_area_labels, area_low_ocr_confidence, no_label_text


class GraphObject(BaseModel):
    """One detected symbol instance (door, window, toilet, ...) from the layout model's icon channels."""
    id: str
    kind: str                          # door | window | toilet | sink | bathtub | closet | sauna_bench | fireplace
    box_px: list[int]
    area_px: int
    conf: float                        # mean icon-class probability over the instance
    room_id: Optional[str] = None


class PlanGraph(BaseModel):
    source: str
    page: int
    image_size: list[int]              # [width, height]
    rooms: list[GraphRoom]
    texts: list[GraphText]
    objects: list[GraphObject] = []
    floors: list[list[int]] = []       # one box per separate building outline (storey drawing)
    min_conf: Union[float, dict[str, float]] = 0.0   # thresholds {texts, rooms, objects}; below them detections were dropped
    dropped: dict[str, int] = {}       # how many texts / rooms / objects were dropped by min_conf
