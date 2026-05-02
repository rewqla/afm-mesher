import unittest

import _bootstrap  # noqa: F401
from src.domain.entities.point import Point
from src.domain.entities.mesh import Mesh
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import (
    distance,
    mesh_average_quality,
    mesh_quality_report,
    orientation,
    point_in_polygon,
    segments_intersect,
    triangle_area,
    triangle_quality,
)


class TestGeometryUtils(unittest.TestCase):
    def test_distance(self) -> None:
        self.assertAlmostEqual(distance(Point(0.0, 0.0), Point(3.0, 4.0)), 5.0)

    def test_orientation(self) -> None:
        a = Point(0.0, 0.0)
        b = Point(1.0, 0.0)
        self.assertEqual(orientation(a, b, Point(1.0, 1.0)), 1)
        self.assertEqual(orientation(a, b, Point(1.0, -1.0)), -1)
        self.assertEqual(orientation(a, b, Point(2.0, 0.0)), 0)

    def test_triangle_area(self) -> None:
        a = Point(0.0, 0.0)
        b = Point(2.0, 0.0)
        c = Point(0.0, 2.0)
        self.assertAlmostEqual(triangle_area(a, b, c), 2.0)
        self.assertAlmostEqual(triangle_area(a, c, b, signed=True), -2.0)

    def test_segments_intersect(self) -> None:
        self.assertTrue(
            segments_intersect(
                Point(0.0, 0.0),
                Point(2.0, 2.0),
                Point(0.0, 2.0),
                Point(2.0, 0.0),
            )
        )
        self.assertFalse(
            segments_intersect(
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(2.0, 0.0),
                Point(3.0, 0.0),
            )
        )
        self.assertTrue(
            segments_intersect(
                Point(0.0, 0.0),
                Point(1.0, 1.0),
                Point(1.0, 1.0),
                Point(2.0, 1.0),
                include_endpoints=True,
            )
        )
        self.assertFalse(
            segments_intersect(
                Point(0.0, 0.0),
                Point(1.0, 1.0),
                Point(1.0, 1.0),
                Point(2.0, 1.0),
                include_endpoints=False,
            )
        )

    def test_point_in_polygon(self) -> None:
        square = [
            Point(0.0, 0.0),
            Point(4.0, 0.0),
            Point(4.0, 4.0),
            Point(0.0, 4.0),
        ]
        self.assertTrue(point_in_polygon(Point(2.0, 2.0), square))
        self.assertFalse(point_in_polygon(Point(5.0, 2.0), square))
        self.assertTrue(point_in_polygon(Point(0.0, 2.0), square, include_boundary=True))
        self.assertFalse(point_in_polygon(Point(0.0, 2.0), square, include_boundary=False))

    def test_triangle_quality(self) -> None:
        a = Point(0.0, 0.0)
        b = Point(1.0, 0.0)
        c = Point(0.5, 0.8660254037844386)
        self.assertAlmostEqual(triangle_quality(a, b, c), 1.0, places=6)

        with self.assertRaises(ValueError):
            triangle_quality(Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0))

    def test_mesh_average_quality(self) -> None:
        mesh = Mesh(
            triangles=[
                Triangle(
                    Point(0.0, 0.0),
                    Point(1.0, 0.0),
                    Point(0.5, 0.8660254037844386),
                ),
                Triangle(
                    Point(0.0, 0.0),
                    Point(1.0, 0.0),
                    Point(0.5, 0.8660254037844386),
                ),
            ]
        )
        self.assertAlmostEqual(mesh_average_quality(mesh), 1.0, places=6)

    def test_mesh_quality_report_histogram(self) -> None:
        mesh = Mesh(
            triangles=[
                Triangle(
                    Point(0.0, 0.0),
                    Point(1.0, 0.0),
                    Point(0.5, 0.8660254037844386),
                ),
                Triangle(
                    Point(0.0, 0.0),
                    Point(1.0, 0.0),
                    Point(0.5, 0.4),
                ),
            ]
        )
        report = mesh_quality_report(mesh, bins=5)
        self.assertGreaterEqual(report.min_quality, 0.0)
        self.assertLessEqual(report.max_quality, 1.0)
        self.assertEqual(len(report.histogram_bins), 6)
        self.assertEqual(len(report.histogram_counts), 5)
        self.assertEqual(sum(report.histogram_counts), 2)


if __name__ == "__main__":
    unittest.main()
