from __future__ import annotations

from src.application.services.region_topology import ClassifiedRegion, RegionPolygon
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import point_in_polygon


class ObstacleProcessor:
    def extract_holes(self, region: ClassifiedRegion) -> list[RegionPolygon]:
        return region.holes

    def filter_triangles_by_holes(self, triangles: list[Triangle], holes: list[RegionPolygon]) -> list[Triangle]:
        if not holes:
            return triangles

        filtered: list[Triangle] = []
        for triangle in triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            if any(point_in_polygon(centroid, hole.points, include_boundary=True) for hole in holes):
                continue
            filtered.append(triangle)
        return filtered

