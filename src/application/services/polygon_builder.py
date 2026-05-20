from __future__ import annotations

from src.application.services.region_topology import RegionPolygon
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import distance


class PolygonBuilder:
    def __init__(self, close_tolerance: float = 1.5) -> None:
        self._close_tolerance = close_tolerance

    def build(self, contour: list[tuple[int, int] | Point]) -> RegionPolygon:
        points = self._normalize_points(contour)
        if len(points) < 3:
            raise ValueError("Contour must contain at least 3 points.")
        if not self._is_closed(points):
            raise ValueError("Contour is not closed.")
        # Snap closure explicitly when the end is within tolerance but not identical.
        if distance(points[0], points[-1]) <= self._close_tolerance and (
            points[0].x != points[-1].x or points[0].y != points[-1].y
        ):
            points = [*points, Point(points[0].x, points[0].y)]

        polygon = points[:-1]
        polygon = self._remove_duplicate_consecutive(polygon)
        if len(polygon) < 3:
            raise ValueError("Contour degenerates after normalization.")
        return RegionPolygon(points=polygon)

    def _normalize_points(self, contour: list[tuple[int, int] | Point]) -> list[Point]:
        points: list[Point] = []
        for item in contour:
            if isinstance(item, Point):
                points.append(Point(float(item.x), float(item.y)))
            else:
                x, y = item
                points.append(Point(float(x), float(y)))
        return points

    def _is_closed(self, points: list[Point]) -> bool:
        if len(points) < 4:
            return False
        return distance(points[0], points[-1]) <= self._close_tolerance

    def _remove_duplicate_consecutive(self, points: list[Point]) -> list[Point]:
        cleaned: list[Point] = []
        for point in points:
            if not cleaned:
                cleaned.append(point)
                continue
            if distance(cleaned[-1], point) > 1e-9:
                cleaned.append(point)
        return cleaned
