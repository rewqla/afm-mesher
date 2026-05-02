import unittest

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import distance, mesh_average_quality

_EPSILON = 1e-6


class TestAfmSubdivisionAndSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(min_triangle_quality=0.01, smoothing_iterations=8)

    def test_subdivide_boundary_respects_target_step(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        h = 20.0
        subdivided = self.mesher.subdivide_boundary(boundary, h)

        self.assertEqual(len(subdivided), 20)

        lengths = [
            distance(subdivided[i], subdivided[(i + 1) % len(subdivided)])
            for i in range(len(subdivided))
        ]
        self.assertTrue(all(length <= h + _EPSILON for length in lengths))
        self.assertTrue(all(abs(length - h) < _EPSILON for length in lengths))

    def test_smoothing_keeps_boundary_fixed_and_improves_quality(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        center_bad = Point(80.0, 50.0)
        mesh = Mesh(
            triangles=[
                Triangle(boundary[0], boundary[1], center_bad),
                Triangle(boundary[1], boundary[2], center_bad),
                Triangle(boundary[2], boundary[3], center_bad),
                Triangle(boundary[3], boundary[0], center_bad),
            ]
        )

        before_quality = mesh_average_quality(mesh)
        smoothed = self.mesher.smooth(mesh, boundary=boundary, iterations=5)
        after_quality = mesh_average_quality(smoothed)

        self.assertGreaterEqual(after_quality, before_quality)

        smoothed_points = {
            (round(t.a.x, 6), round(t.a.y, 6))
            for t in smoothed.triangles
        }
        smoothed_points.update((round(t.b.x, 6), round(t.b.y, 6)) for t in smoothed.triangles)
        smoothed_points.update((round(t.c.x, 6), round(t.c.y, 6)) for t in smoothed.triangles)

        for boundary_point in boundary:
            self.assertIn((round(boundary_point.x, 6), round(boundary_point.y, 6)), smoothed_points)

        internal_points = [
            point for point in smoothed_points
            if point not in {(p.x, p.y) for p in boundary}
        ]
        self.assertEqual(len(internal_points), 1)
        internal_x, internal_y = internal_points[0]
        self.assertLess(abs(internal_x - 50.0), abs(center_bad.x - 50.0))
        self.assertLessEqual(abs(internal_y - 50.0), abs(center_bad.y - 50.0))


if __name__ == "__main__":
    unittest.main()
