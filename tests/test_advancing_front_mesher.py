import unittest

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import point_in_polygon


class TestAdvancingFrontMesher(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(min_triangle_quality=0.01, smoothing_iterations=6)

    def test_subdivide_boundary_uniform(self) -> None:
        square = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ]
        subdivided = self.mesher.subdivide_boundary(square, h=2.0)
        self.assertEqual(len(subdivided), 20)

    def test_generate_mesh_for_square_boundary(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ]

        mesh = self.mesher.generate(boundary)
        self.assertGreaterEqual(len(mesh.triangles), 2)
        self._assert_triangles_inside_polygon(mesh, boundary)

    def test_generate_mesh_for_concave_l_shape_boundary(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(8.0, 0.0),
            Point(8.0, 3.0),
            Point(3.0, 3.0),
            Point(3.0, 8.0),
            Point(0.0, 8.0),
        ]

        mesh = self.mesher.generate(boundary)
        self.assertGreaterEqual(len(mesh.triangles), 3)
        self._assert_triangles_inside_polygon(mesh, boundary)

    def test_rejects_degenerate_boundary(self) -> None:
        boundary = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
        with self.assertRaises(ValueError):
            self.mesher.generate(boundary)

    def _assert_triangles_inside_polygon(self, mesh, polygon: list[Point]) -> None:
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))


if __name__ == "__main__":
    unittest.main()
