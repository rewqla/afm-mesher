from __future__ import annotations

from src.application.services.polygon_builder import PolygonBuilder
from src.application.services.region_topology import RegionPolygon


class BoundaryDetectionService:
    def __init__(self, polygon_builder: PolygonBuilder | None = None) -> None:
        self._polygon_builder = polygon_builder or PolygonBuilder()

    def detect(self, contours: list[list[tuple[int, int]]]) -> list[RegionPolygon]:
        boundaries: list[RegionPolygon] = []
        for contour in contours:
            try:
                boundaries.append(self._polygon_builder.build(contour))
            except ValueError:
                continue
        return boundaries

