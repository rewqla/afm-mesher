import unittest
from itertools import combinations

import _bootstrap  # noqa: F401
from src.application.dto.complex_test_contours import (
    hourglass_contour,
    star_contour,
    u_shape_contour,
)
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import point_in_polygon, segments_intersect

_ROUND = 8
_EPSILON = 1e-6


class TestAfmMeshIntegrityExtra(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            smoothing_iterations=0,  # exact topology/area checks before smoothing
        )

    def test_area_conservation(self) -> None:
        for name, boundary in self._boundaries().items():
            with self.subTest(shape=name):
                mesh = self.mesher.generate(boundary)
                polygon_area = self._polygon_area(boundary)
                mesh_area = sum(t.area() for t in mesh.triangles)
                self.assertAlmostEqual(mesh_area, polygon_area, delta=1e-6)

    def test_edge_usage_and_euler_characteristic(self) -> None:
        for name, boundary in self._boundaries().items():
            with self.subTest(shape=name):
                mesh = self.mesher.generate(boundary)
                edge_counts = self._edge_usage(mesh)

                # Every edge is either boundary (1) or internal (2).
                self.assertTrue(all(count in (1, 2) for count in edge_counts.values()))
                self.assertGreaterEqual(sum(1 for c in edge_counts.values() if c == 1), 3)
                self.assertGreaterEqual(sum(1 for c in edge_counts.values() if c == 2), 1)

                v_count = len(self._unique_vertices(mesh))
                e_count = len(edge_counts)
                f_count = len(mesh.triangles)
                # Connected planar triangulation of one region should satisfy Euler: V - E + F = 1.
                self.assertEqual(v_count - e_count + f_count, 1)

    def test_all_mesh_nodes_within_input_polygon(self) -> None:
        for name, boundary in self._boundaries().items():
            with self.subTest(shape=name):
                mesh = self.mesher.generate(boundary)
                for p in self._unique_vertices(mesh):
                    self.assertTrue(point_in_polygon(p, boundary, include_boundary=True))

    def test_no_triangle_self_intersections(self) -> None:
        for name, boundary in self._boundaries().items():
            with self.subTest(shape=name):
                mesh = self.mesher.generate(boundary)
                triangles = mesh.triangles

                for t1, t2 in combinations(triangles, 2):
                    # Pairs sharing vertices/edges are allowed.
                    t1_points = {t1.a, t1.b, t1.c}
                    t2_points = {t2.a, t2.b, t2.c}
                    if t1_points & t2_points:
                        continue

                    for e1 in self._triangle_edges(t1):
                        for e2 in self._triangle_edges(t2):
                            self.assertFalse(
                                segments_intersect(e1[0], e1[1], e2[0], e2[1], include_endpoints=False),
                                msg=f"Unexpected intersection: {e1} vs {e2}",
                            )

    def _boundaries(self) -> dict[str, list[Point]]:
        return {
            "star": star_contour(),
            "u_shape": u_shape_contour(),
            "hourglass": hourglass_contour(),
        }

    def _polygon_area(self, polygon: list[Point]) -> float:
        area = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y
        return abs(area) / 2.0

    def _triangle_edges(self, triangle) -> list[tuple[Point, Point]]:
        return [(triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)]

    def _edge_usage(self, mesh) -> dict[tuple[tuple[float, float], tuple[float, float]], int]:
        edge_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
        for triangle in mesh.triangles:
            for a, b in self._triangle_edges(triangle):
                key = self._edge_key(a, b)
                edge_counts[key] = edge_counts.get(key, 0) + 1
        return edge_counts

    def _unique_vertices(self, mesh) -> set[Point]:
        vertices: set[Point] = set()
        for triangle in mesh.triangles:
            vertices.add(triangle.a)
            vertices.add(triangle.b)
            vertices.add(triangle.c)
        return vertices

    def _edge_key(self, a: Point, b: Point) -> tuple[tuple[float, float], tuple[float, float]]:
        pa = (round(a.x, _ROUND), round(a.y, _ROUND))
        pb = (round(b.x, _ROUND), round(b.y, _ROUND))
        return (pa, pb) if pa <= pb else (pb, pa)


if __name__ == "__main__":
    unittest.main()
