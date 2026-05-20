from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities.point import Point


@dataclass(frozen=True, slots=True)
class RegionPolygon:
    points: list[Point]


@dataclass(frozen=True, slots=True)
class ClassifiedRegion:
    shell: RegionPolygon
    holes: list[RegionPolygon]

