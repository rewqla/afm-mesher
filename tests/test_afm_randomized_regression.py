import math
import random
import unittest

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import mesh_average_quality, point_in_polygon


class TestAfmRandomizedRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            smoothing_iterations=8,
        )

    def test_randomized_convex_polygons(self) -> None:
        for seed in range(6):
            with self.subTest(seed=seed):
                boundary = self._make_random_convex_polygon(seed, count=10)
                mesh = self.mesher.generate(boundary)

                self.assertGreater(len(mesh.triangles), 0)
                self.assertGreater(mesh_average_quality(mesh), 0.45)
                self._assert_centroids_inside(mesh, boundary)
                self._assert_positive_area(mesh)

    def _make_random_convex_polygon(self, seed: int, count: int) -> list[Point]:
        rng = random.Random(seed)
        center_x, center_y = 250.0, 250.0
        angles = sorted(rng.uniform(0.0, 360.0) for _ in range(count))
        points: list[Point] = []
        for angle_deg in angles:
            angle = math.radians(angle_deg)
            radius = rng.uniform(110.0, 190.0)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append(Point(x, y))
        return points

    def _assert_centroids_inside(self, mesh, polygon: list[Point]) -> None:
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))

    def _assert_positive_area(self, mesh) -> None:
        for triangle in mesh.triangles:
            self.assertGreater(triangle.area(), 0.0)


if __name__ == "__main__":
    unittest.main()
