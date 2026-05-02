from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    min_quality: float
    mean_quality: float
    max_quality: float
    histogram_bins: list[float]
    histogram_counts: list[int]


def distance(a: Point, b: Point) -> float:
    return sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)


def orientation(a: Point, b: Point, c: Point, epsilon: float = _EPSILON) -> int:
    cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    if abs(cross) <= epsilon:
        return 0
    return 1 if cross > 0 else -1


def triangle_area(a: Point, b: Point, c: Point, signed: bool = False) -> float:
    area2 = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2.0
    return area2 if signed else abs(area2)


def segments_intersect(
    p1: Point,
    p2: Point,
    q1: Point,
    q2: Point,
    include_endpoints: bool = True,
) -> bool:
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if not include_endpoints:
        return o1 * o2 < 0 and o3 * o4 < 0

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True

    return False


def point_in_polygon(point: Point, polygon: list[Point], include_boundary: bool = True) -> bool:
    if len(polygon) < 3:
        raise ValueError("Polygon must contain at least 3 vertices.")

    inside = False
    n = len(polygon)

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]

        if orientation(a, point, b) == 0 and _on_segment(a, point, b):
            return include_boundary

        intersects = ((a.y > point.y) != (b.y > point.y))
        if intersects:
            x_intersection = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
            if x_intersection > point.x:
                inside = not inside

    return inside


def triangle_quality(a: Point, b: Point, c: Point) -> float:
    ab = distance(a, b)
    bc = distance(b, c)
    ca = distance(c, a)
    area = triangle_area(a, b, c)

    if area <= _EPSILON:
        raise ValueError("Triangle quality is undefined for degenerate triangles.")

    denominator = ab**2 + bc**2 + ca**2
    quality = 4.0 * sqrt(3.0) * area / denominator
    return max(0.0, min(1.0, quality))


def mesh_average_quality(mesh: Mesh) -> float:
    if not mesh.triangles:
        raise ValueError("Mesh must contain at least one triangle.")
    qualities = _mesh_quality_values(mesh)
    return sum(qualities) / len(qualities)


def mesh_quality_report(mesh: Mesh, bins: int = 10) -> MeshQualityReport:
    if bins < 1:
        raise ValueError("bins must be >= 1")
    qualities = _mesh_quality_values(mesh)
    min_quality = min(qualities)
    mean_quality = sum(qualities) / len(qualities)
    max_quality = max(qualities)

    edges = [i / bins for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for quality in qualities:
        idx = min(int(quality * bins), bins - 1)
        counts[idx] += 1

    return MeshQualityReport(
        min_quality=min_quality,
        mean_quality=mean_quality,
        max_quality=max_quality,
        histogram_bins=edges,
        histogram_counts=counts,
    )


def _mesh_quality_values(mesh: Mesh) -> list[float]:
    if not mesh.triangles:
        raise ValueError("Mesh must contain at least one triangle.")
    return [triangle_quality(triangle.a, triangle.b, triangle.c) for triangle in mesh.triangles]


def _on_segment(a: Point, p: Point, b: Point) -> bool:
    return (
        min(a.x, b.x) - _EPSILON <= p.x <= max(a.x, b.x) + _EPSILON
        and min(a.y, b.y) - _EPSILON <= p.y <= max(a.y, b.y) + _EPSILON
    )
