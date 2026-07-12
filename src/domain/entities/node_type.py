from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    INTERFACE = "interface"
