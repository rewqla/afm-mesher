from __future__ import annotations

from enum import Enum


class Tool(str, Enum):
    PEN = "pen"
    ERASER = "eraser"
    FILL = "fill"
    POINT = "point"
    SEGMENT = "segment"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"


class TriangulationMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    CUSTOM = "custom"
