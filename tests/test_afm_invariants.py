from itertools import combinations
import unittest

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import orientation, point_in_polygon, segments_intersect

_EPSILON = 1e-9


class TestAfmInvariants(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(min_triangle_quality=0.01)

    def test_invariants_for_square_and_concave_polygon(self) -> None:
        boundaries = {
            "square": [
                Point(0.0, 0.0),
                Point(10.0, 0.0),
                Point(10.0, 10.0),
                Point(0.0, 10.0),
            ],
            "concave_l_shape": [
                Point(0.0, 0.0),
                Point(8.0, 0.0),
                Point(8.0, 3.0),
                Point(3.0, 3.0),
                Point(3.0, 8.0),
                Point(0.0, 8.0),
            ],
        }

        for name, boundary in boundaries.items():
            with self.subTest(boundary=name):
                mesh = self.mesher.generate(boundary)
                self.assertGreater(len(mesh.triangles), 0)
                self._assert_positive_triangle_areas(mesh.triangles)
                self._assert_triangles_inside_polygon(mesh.triangles, boundary)
                self._assert_no_self_intersections(mesh.triangles)

    def _assert_positive_triangle_areas(self, triangles) -> None:
        for triangle in triangles:
            self.assertGreater(triangle.area(), 0.0)

    def _assert_triangles_inside_polygon(self, triangles, polygon: list[Point]) -> None:
        for triangle in triangles:
            for vertex in (triangle.a, triangle.b, triangle.c):
                self.assertTrue(point_in_polygon(vertex, polygon, include_boundary=True))

            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))

    def _assert_no_self_intersections(self, triangles) -> None:
        edges = []
        for triangle in triangles:
            edges.extend(
                [
                    (triangle.a, triangle.b),
                    (triangle.b, triangle.c),
                    (triangle.c, triangle.a),
                ]
            )

        for edge_a, edge_b in combinations(edges, 2):
            if self._share_endpoint(edge_a, edge_b):
                continue

            self.assertFalse(
                segments_intersect(
                    edge_a[0],
                    edge_a[1],
                    edge_b[0],
                    edge_b[1],
                    include_endpoints=False,
                ),
                msg=f"Intersecting edges found: {edge_a} vs {edge_b}",
            )

            self.assertFalse(
                self._has_collinear_overlap(edge_a, edge_b),
                msg=f"Collinear overlapping edges found: {edge_a} vs {edge_b}",
            )

    def _share_endpoint(self, edge_a, edge_b) -> bool:
        return (
            edge_a[0] == edge_b[0]
            or edge_a[0] == edge_b[1]
            or edge_a[1] == edge_b[0]
            or edge_a[1] == edge_b[1]
        )

    def _has_collinear_overlap(self, edge_a, edge_b) -> bool:
        a, b = edge_a
        c, d = edge_b
        if orientation(a, b, c) != 0 or orientation(a, b, d) != 0:
            return False

        use_x_axis = abs(a.x - b.x) >= abs(a.y - b.y)
        if use_x_axis:
            a_min, a_max = sorted((a.x, b.x))
            b_min, b_max = sorted((c.x, d.x))
        else:
            a_min, a_max = sorted((a.y, b.y))
            b_min, b_max = sorted((c.y, d.y))

        overlap = min(a_max, b_max) - max(a_min, b_min)
        return overlap > _EPSILON


if __name__ == "__main__":
    unittest.main()
