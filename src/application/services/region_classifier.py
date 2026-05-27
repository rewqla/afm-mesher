from __future__ import annotations

from src.application.services.region_topology import ClassifiedRegion, RegionPolygon
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import point_in_polygon


class RegionClassifier:
    def classify(self, polygons: list[RegionPolygon]) -> list[ClassifiedRegion]:
        if not polygons:
            return []

        sorted_polygons = sorted(polygons, key=lambda p: abs(self._signed_area(p.points)), reverse=True)
        parent_idx: dict[int, int | None] = {}
        for idx, polygon in enumerate(sorted_polygons):
            parent_idx[idx] = self._find_parent(idx, polygon, sorted_polygons)

        children: dict[int, list[int]] = {idx: [] for idx in range(len(sorted_polygons))}
        for idx, parent in parent_idx.items():
            if parent is not None:
                children[parent].append(idx)

        regions: list[ClassifiedRegion] = []
        for idx, polygon in enumerate(sorted_polygons):
            depth = self._depth(idx, parent_idx)
            if depth % 2 != 0:
                continue
            holes = [sorted_polygons[child] for child in children[idx]]
            regions.append(ClassifiedRegion(shell=polygon, holes=holes))
        return regions

    def _find_parent(self, idx: int, polygon: RegionPolygon, polygons: list[RegionPolygon]) -> int | None:
        candidate_point = self._representative_point(polygon.points)
        parent: int | None = None
        for candidate_idx in range(idx):
            candidate_polygon = polygons[candidate_idx]
            if point_in_polygon(candidate_point, candidate_polygon.points, include_boundary=True):
                parent = candidate_idx
        return parent

    def _representative_point(self, points: list[Point]) -> Point:
        # Avoid using a boundary vertex as representative point.
        centroid = self._polygon_centroid(points)
        if point_in_polygon(centroid, points, include_boundary=False):
            return centroid

        avg_x = sum(point.x for point in points) / len(points)
        avg_y = sum(point.y for point in points) / len(points)
        average = Point(avg_x, avg_y)
        if point_in_polygon(average, points, include_boundary=False):
            return average

        for point in points:
            probe = Point((point.x + centroid.x) * 0.5, (point.y + centroid.y) * 0.5)
            if point_in_polygon(probe, points, include_boundary=False):
                return probe

        # Fallback for degenerate/near-degenerate shapes.
        return points[0]

    def _depth(self, idx: int, parent_idx: dict[int, int | None]) -> int:
        depth = 0
        current = parent_idx[idx]
        while current is not None:
            depth += 1
            current = parent_idx[current]
        return depth

    def _signed_area(self, polygon: list[Point]) -> float:
        area = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y
        return area / 2.0

    def _polygon_centroid(self, polygon: list[Point]) -> Point:
        area2 = 0.0
        cx_acc = 0.0
        cy_acc = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            cross = p1.x * p2.y - p2.x * p1.y
            area2 += cross
            cx_acc += (p1.x + p2.x) * cross
            cy_acc += (p1.y + p2.y) * cross

        if abs(area2) <= 1e-12:
            avg_x = sum(point.x for point in polygon) / max(1, len(polygon))
            avg_y = sum(point.y for point in polygon) / max(1, len(polygon))
            return Point(avg_x, avg_y)

        factor = 1.0 / (3.0 * area2)
        return Point(cx_acc * factor, cy_acc * factor)
