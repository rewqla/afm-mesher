from __future__ import annotations

from src.application.services.region_topology import ClassifiedRegion, RegionPolygon
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import orientation, point_in_polygon, segments_intersect
from src.infrastructure.processing.image_boundary_extractor import extract_contours
from src.infrastructure.image.photo_preprocessor import simplify_contour

from math import ceil, floor
from PIL import Image, ImageDraw

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.geometry.polygon import orient as orient_polygon
    from shapely.ops import unary_union

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover - optional dependency path
    GeometryCollection = MultiPolygon = Polygon = None  # type: ignore[assignment]
    orient_polygon = unary_union = None  # type: ignore[assignment]
    _HAS_SHAPELY = False


class ObstacleProcessor:
    def extract_holes(self, region: ClassifiedRegion) -> list[RegionPolygon]:
        return region.holes

    def normalize_holes(self, shell: RegionPolygon, holes: list[RegionPolygon]) -> list[RegionPolygon]:
        if not holes:
            return []
        holes = self._filter_micro_holes(shell, holes)
        if not holes:
            return []
        if len(holes) == 1:
            return [holes[0]]
        if not self._holes_need_union(holes):
            return sorted(holes, key=lambda polygon: abs(self._signed_area(polygon.points)), reverse=True)

        if _HAS_SHAPELY:
            normalized = self._normalize_holes_with_shapely(shell, holes)
            if normalized:
                return sorted(normalized, key=lambda polygon: abs(self._signed_area(polygon.points)), reverse=True)

        # The editor works in pixel geometry, so we merge overlapping/touching holes
        # by rasterizing them into a temporary mask and extracting the merged contour set.
        bounds = self._raster_bounds([shell.points, *[hole.points for hole in holes]])
        mask = self._rasterize_holes(holes, bounds)
        contours = extract_contours(mask)

        normalized: list[RegionPolygon] = []
        for contour in contours:
            simplified = simplify_contour(contour, epsilon=1.0)
            points = self._shift_contour(simplified, bounds[0], bounds[1])
            points = self._remove_duplicate_consecutive(points)
            if len(points) < 3:
                continue
            normalized.append(RegionPolygon(points=points))

        if not normalized:
            raise ValueError("Failed to normalize overlapping holes.")

        return sorted(normalized, key=lambda polygon: abs(self._signed_area(polygon.points)), reverse=True)

    def _normalize_holes_with_shapely(self, shell: RegionPolygon, holes: list[RegionPolygon]) -> list[RegionPolygon]:
        if not _HAS_SHAPELY:
            return []

        shell_polygon = self._to_shapely_polygon(shell.points)
        if shell_polygon is None:
            return []
        shell_polygon = shell_polygon.buffer(0)
        if shell_polygon.is_empty:
            return []

        hole_polygons = [self._to_shapely_polygon(hole.points) for hole in holes]
        hole_polygons = [polygon.buffer(0) for polygon in hole_polygons if polygon is not None]
        hole_polygons = [polygon for polygon in hole_polygons if not polygon.is_empty and polygon.area > 0.0]
        if not hole_polygons:
            return []

        merged = unary_union(hole_polygons)
        if merged.is_empty:
            return []

        # Merge contours that only touch at a point or have tiny gaps from raster/input noise.
        tolerance = self._merge_tolerance(shell.points, holes)
        if tolerance > 0.0:
            merged = merged.buffer(tolerance, join_style=2).buffer(-tolerance, join_style=2)
            merged = merged.buffer(0)
        if merged.is_empty:
            return []

        clipped = merged.intersection(shell_polygon)
        clipped = clipped.buffer(0)
        if clipped.is_empty:
            return []

        polygons = self._extract_polygons(clipped)
        if not polygons:
            return []

        normalized: list[RegionPolygon] = []
        for polygon in polygons:
            if polygon.is_empty or polygon.area <= 0.0:
                continue
            oriented = orient_polygon(polygon, sign=-1.0)
            points = self._ring_to_points(list(oriented.exterior.coords))
            if len(points) < 3:
                continue
            normalized.append(RegionPolygon(points=points))
        return normalized

    def filter_triangles_by_holes(self, triangles: list[Triangle], holes: list[RegionPolygon]) -> list[Triangle]:
        if not holes:
            return triangles

        filtered: list[Triangle] = []
        for triangle in triangles:
            if any(self._triangle_touches_hole(triangle, hole) for hole in holes):
                continue
            filtered.append(triangle)
        return filtered

    def _triangle_touches_hole(self, triangle: Triangle, hole: RegionPolygon) -> bool:
        triangle_points = (triangle.a, triangle.b, triangle.c)
        centroid = Point(
            (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
            (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
        )
        if point_in_polygon(centroid, hole.points, include_boundary=True):
            return True
        if any(point_in_polygon(point, hole.points, include_boundary=True) for point in triangle_points):
            return True

        triangle_edges = (
            (triangle.a, triangle.b),
            (triangle.b, triangle.c),
            (triangle.c, triangle.a),
        )
        hole_edges = self._polygon_edges(hole.points)
        for triangle_edge in triangle_edges:
            for hole_edge in hole_edges:
                if segments_intersect(triangle_edge[0], triangle_edge[1], hole_edge[0], hole_edge[1]):
                    return True

        return any(self._point_in_triangle(point, triangle) for point in hole.points[:-1])

    def _polygon_edges(self, points: list[Point]) -> list[tuple[Point, Point]]:
        if len(points) < 2:
            return []
        edges = [(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]
        if points[0] != points[-1]:
            edges.append((points[-1], points[0]))
        return edges

    def _point_in_triangle(self, point: Point, triangle: Triangle) -> bool:
        o1 = orientation(triangle.a, triangle.b, point)
        o2 = orientation(triangle.b, triangle.c, point)
        o3 = orientation(triangle.c, triangle.a, point)
        return (o1 >= 0 and o2 >= 0 and o3 >= 0) or (o1 <= 0 and o2 <= 0 and o3 <= 0)

    def _raster_bounds(self, polygon_groups: list[list[Point]]) -> tuple[int, int, int, int]:
        xs = [point.x for group in polygon_groups for point in group]
        ys = [point.y for group in polygon_groups for point in group]
        min_x = floor(min(xs)) - 1
        min_y = floor(min(ys)) - 1
        max_x = ceil(max(xs)) + 1
        max_y = ceil(max(ys)) + 1
        return min_x, min_y, max_x, max_y

    def _holes_need_union(self, holes: list[RegionPolygon]) -> bool:
        for i, left in enumerate(holes):
            left_bounds = self._polygon_bounds(left.points)
            for right in holes[i + 1:]:
                if not self._bounds_overlap(left_bounds, self._polygon_bounds(right.points)):
                    continue
                if self._polygons_touch_or_overlap(left.points, right.points):
                    return True
        return False

    def _polygon_bounds(self, points: list[Point]) -> tuple[float, float, float, float]:
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _bounds_overlap(
        self,
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> bool:
        left_min_x, left_min_y, left_max_x, left_max_y = left
        right_min_x, right_min_y, right_max_x, right_max_y = right
        return not (
            left_max_x < right_min_x
            or right_max_x < left_min_x
            or left_max_y < right_min_y
            or right_max_y < left_min_y
        )

    def _polygons_touch_or_overlap(self, left: list[Point], right: list[Point]) -> bool:
        left_edges = self._polygon_edges(left)
        right_edges = self._polygon_edges(right)

        for edge_left in left_edges:
            for edge_right in right_edges:
                if segments_intersect(
                    edge_left[0],
                    edge_left[1],
                    edge_right[0],
                    edge_right[1],
                    include_endpoints=True,
                ):
                    return True

        if left and point_in_polygon(left[0], right, include_boundary=True):
            return True
        if right and point_in_polygon(right[0], left, include_boundary=True):
            return True
        return False

    def _rasterize_holes(self, holes: list[RegionPolygon], bounds: tuple[int, int, int, int]) -> list[list[int]]:
        min_x, min_y, max_x, max_y = bounds
        width = max(1, max_x - min_x + 1)
        height = max(1, max_y - min_y + 1)

        image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(image)

        for hole in holes:
            polygon = [(point.x - min_x, point.y - min_y) for point in hole.points]
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=1, outline=1)

        mask: list[list[int]] = []
        for y in range(height):
            row: list[int] = []
            for x in range(width):
                row.append(1 if image.getpixel((x, y)) else 0)
            mask.append(row)
        return mask

    def _shift_contour(self, contour: list[tuple[int, int]], min_x: int, min_y: int) -> list[Point]:
        return [Point(float(x + min_x), float(y + min_y)) for x, y in contour]

    def _remove_duplicate_consecutive(self, points: list[Point]) -> list[Point]:
        cleaned: list[Point] = []
        for point in points:
            if not cleaned or cleaned[-1] != point:
                cleaned.append(point)
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        return cleaned

    def _signed_area(self, points: list[Point]) -> float:
        area = 0.0
        n = len(points)
        for idx in range(n):
            p1 = points[idx]
            p2 = points[(idx + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y
        return area / 2.0

    def _to_shapely_polygon(self, points: list[Point]) -> Polygon | None:
        if not _HAS_SHAPELY:
            return None
        unique = self._remove_duplicate_consecutive(points)
        if len(unique) < 3:
            return None
        coords = [(point.x, point.y) for point in unique]
        return Polygon(coords)

    def _merge_tolerance(self, shell_points: list[Point], holes: list[RegionPolygon]) -> float:
        min_x = min(point.x for point in shell_points)
        max_x = max(point.x for point in shell_points)
        min_y = min(point.y for point in shell_points)
        max_y = max(point.y for point in shell_points)
        domain_scale = max(max_x - min_x, max_y - min_y, 1.0)
        tolerance = domain_scale * 1e-6

        min_edge = float("inf")
        for hole in holes:
            hole_points = self._remove_duplicate_consecutive(hole.points)
            for idx in range(len(hole_points)):
                p1 = hole_points[idx]
                p2 = hole_points[(idx + 1) % len(hole_points)]
                edge = ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
                if edge > 0.0:
                    min_edge = min(min_edge, edge)
        if min_edge != float("inf"):
            tolerance = max(tolerance, min_edge * 1e-4)
        return tolerance

    def _extract_polygons(self, geometry: Polygon | MultiPolygon | GeometryCollection) -> list[Polygon]:
        if not _HAS_SHAPELY:
            return []
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, MultiPolygon):
            return [poly for poly in geometry.geoms if isinstance(poly, Polygon)]
        if isinstance(geometry, GeometryCollection):
            polygons: list[Polygon] = []
            for geom in geometry.geoms:
                polygons.extend(self._extract_polygons(geom))
            return polygons
        return []

    def _ring_to_points(self, coords: list[tuple[float, float]]) -> list[Point]:
        points: list[Point] = [Point(float(x), float(y)) for x, y in coords[:-1]]
        return self._remove_duplicate_consecutive(points)

    def _filter_micro_holes(self, shell: RegionPolygon, holes: list[RegionPolygon]) -> list[RegionPolygon]:
        shell_area = abs(self._signed_area(shell.points))
        if shell_area <= 0.0:
            return holes

        # Reject tiny artifacts from antialiasing/mesh overlays while keeping intended obstacles.
        min_area = max(4.0, shell_area * 1e-4)
        filtered = [hole for hole in holes if abs(self._signed_area(hole.points)) >= min_area and len(hole.points) >= 3]
        return filtered
