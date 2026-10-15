from __future__ import annotations

from src.application.services.region_topology import ClassifiedRegion, RegionPolygon
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import point_in_polygon


class RegionClassifier:
    def classify(self, polygons: list[RegionPolygon]) -> list[ClassifiedRegion]:
        if not polygons:
            return []

        sorted_polygons = sorted(polygons, key=lambda p: abs(self._signed_area(p.points)), reverse=True)
        self._bounds_cache: dict[int, tuple[list[Point], tuple[float, float, float, float]]] = {}
        parent_idx: dict[int, int | None] = {}
        bounds = [self._bounds(polygon.points) for polygon in sorted_polygons]
        for idx, polygon in enumerate(sorted_polygons):
            parent_idx[idx] = self._find_parent(idx, polygon, sorted_polygons, bounds)

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

    def _find_parent(
            self,
            idx: int,
            polygon: RegionPolygon,
            polygons: list[RegionPolygon],
            bounds: list[tuple[float, float, float, float]] | None = None,
    ) -> int | None:
        candidate_point = self._representative_point(polygon.points)
        px = candidate_point.x
        py = candidate_point.y
        parent: int | None = None
        for candidate_idx in range(idx):
            candidate_polygon = polygons[candidate_idx]
            # A point outside the (slightly widened) bounding box is neither inside nor on the boundary.
            x0, x1, y0, y1 = bounds[candidate_idx] if bounds is not None else self._bounds(candidate_polygon.points)
            if not (x0 <= px <= x1 and y0 <= py <= y1) and len(candidate_polygon.points) >= 3:
                continue
            if point_in_polygon(candidate_point, candidate_polygon.points, include_boundary=True):
                parent = candidate_idx
        return parent

    def _bounds(self, points: list[Point]) -> tuple[float, float, float, float]:
        cache = self.__dict__.setdefault("_bounds_cache", {})
        cached = cache.get(id(points))
        if cached is None or cached[0] is not points:
            xs = [p.x for p in points] or [0.0]
            ys = [p.y for p in points] or [0.0]
            cached = (points, (min(xs) - 1e-6, max(xs) + 1e-6, min(ys) - 1e-6, max(ys) + 1e-6))
            cache[id(points)] = cached
        return cached[1]

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
