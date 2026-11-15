"""Offline test of the PDF geometry helpers and the schema filling (no PDF file needed).

Run: python tests/test_pdf_plan_offline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pipelines.pdf_plan import (
    Region,
    TextLine,
    regions_of,
    room_regions,
)


def square(x0: float, y0: float, size: float) -> np.ndarray:
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]], float)


def test_regions() -> None:
    # a room with a column hole: 10x10 minus 2x2
    room = regions_of([square(0, 0, 10), square(4, 4, 2)])
    assert len(room) == 1 and room[0].area == 96.0
    assert room[0].contains(1, 1) and not room[0].contains(5, 5)
    # walls drawn as one path of separate pieces: separate regions, never one big polygon
    walls = regions_of([square(0, 0, 1), square(20, 0, 1)])
    assert len(walls) == 2 and not any(w.contains(10, 0.5) for w in walls)


def test_room_style() -> None:
    room_style, fixture_style = (((138, 138, 138), 0.37)), (((0, 0, 0), 0.51))
    rooms = [Region(square(0, 0, 10), (), room_style), Region(square(20, 0, 10), (), room_style)]
    bathtub = Region(square(1, 1, 2), (), fixture_style)  # fixture symbol under room 1's number
    numbers = [TextLine("1.01", 2, 2), TextLine("1.02", 25, 5), TextLine("1.03", 21, 1)]
    kept = room_regions(rooms + [bathtub], numbers + [TextLine("1.04", 5, 5)])
    assert bathtub not in kept and len(kept) == 2


if __name__ == "__main__":
    test_regions()
    test_room_style()
    print("OK")
