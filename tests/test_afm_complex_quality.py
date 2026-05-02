import unittest

import _bootstrap  # noqa: F401
from src.application.dto.complex_test_contours import (
    hourglass_contour,
    star_contour,
    u_shape_contour,
)
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import mesh_average_quality, point_in_polygon


class TestAfmComplexQuality(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            smoothing_iterations=8,
        )

    def test_complex_shapes_quality_target(self) -> None:
        boundaries = {
            "star": star_contour(),
            "u_shape": u_shape_contour(),
            "hourglass": hourglass_contour(),
        }

        for name, boundary in boundaries.items():
            with self.subTest(shape=name):
                mesh = self.mesher.generate(boundary)
                avg_quality = mesh_average_quality(mesh)

                self.assertGreater(len(mesh.triangles), 0)
                self.assertGreaterEqual(
                    avg_quality,
                    0.7,
                    msg=f"{name} avg_quality is too low: {avg_quality:.3f}",
                )
                self._assert_triangles_inside_polygon(mesh.triangles, boundary)

    def _assert_triangles_inside_polygon(self, triangles, polygon: list[Point]) -> None:
        for triangle in triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))


if __name__ == "__main__":
    unittest.main()
