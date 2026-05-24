import unittest

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import point_in_polygon, segments_intersect


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

    def test_generate_mesh_with_hole_starts_front_from_hole_boundary(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=50.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        hole = [
            Point(40.0, 40.0),
            Point(60.0, 40.0),
            Point(60.0, 60.0),
            Point(40.0, 60.0),
        ]

        mesh = mesher.generate_with_holes(boundary, [hole])

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        self._assert_triangles_outside_hole(mesh, hole)
        self._assert_hole_boundary_is_used_by_mesh(mesh, hole)

    def test_generate_mesh_with_multiple_holes_starts_front_from_every_hole(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=60.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(160.0, 0.0),
            Point(160.0, 100.0),
            Point(0.0, 100.0),
        ]
        holes = [
            [
                Point(25.0, 25.0),
                Point(45.0, 25.0),
                Point(45.0, 45.0),
                Point(25.0, 45.0),
            ],
            [
                Point(70.0, 45.0),
                Point(95.0, 45.0),
                Point(95.0, 70.0),
                Point(70.0, 70.0),
            ],
            [
                Point(120.0, 20.0),
                Point(140.0, 20.0),
                Point(140.0, 40.0),
                Point(120.0, 40.0),
            ],
        ]

        mesh = mesher.generate_with_holes(boundary, holes)

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        for hole in holes:
            self._assert_triangles_outside_hole(mesh, hole)
            self._assert_hole_boundary_is_used_by_mesh(mesh, hole)

    def test_prepare_cut_boundary_builds_closed_internal_barrier(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        cut = [Point(30.0, 60.0), Point(90.0, 60.0)]

        barrier = self.mesher._prepare_cut_boundary(cut, boundary)

        self.assertGreaterEqual(len(barrier), 4)
        self.assertGreater(abs(self.mesher._signed_area(barrier)), 0.0)
        self.assertNotEqual(barrier[0], barrier[-1])

    def test_generate_with_holes_and_cuts_accepts_open_line_as_internal_barrier(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=50.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        cut = [Point(20.0, 50.0), Point(80.0, 50.0)]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[cut])

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        self.assertTrue(self._mesh_respects_cut_line(mesh, cut))

    def test_rejects_degenerate_boundary(self) -> None:
        boundary = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
        with self.assertRaises(ValueError):
            self.mesher.generate(boundary)

    def test_build_topology_uses_global_linear_triangle_node_numbers(self) -> None:
        mesh = Mesh(
            triangles=[
                Triangle(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                Triangle(Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
            ]
        )
        boundary = [Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)]

        positions, triangles, _, _, _ = self.mesher._build_topology(mesh, boundary)

        self.assertEqual(sorted(positions.keys()), [1, 2, 3, 4])
        self.assertEqual(triangles[0][1], triangles[1][0])
        self.assertEqual(triangles[0][2], triangles[1][2])

    def _assert_triangles_inside_polygon(self, mesh, polygon: list[Point]) -> None:
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))

    def _assert_triangles_outside_hole(self, mesh, hole: list[Point]) -> None:
        hole_edges = [(hole[idx], hole[(idx + 1) % len(hole)]) for idx in range(len(hole))]
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertFalse(point_in_polygon(centroid, hole, include_boundary=False))
            triangle_edges = ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
            for edge in triangle_edges:
                for hole_edge in hole_edges:
                    if self._same_undirected_edge(edge, hole_edge):
                        continue
                    self.assertFalse(
                        segments_intersect(edge[0], edge[1], hole_edge[0], hole_edge[1], include_endpoints=False)
                    )

    def _assert_hole_boundary_is_used_by_mesh(self, mesh, hole: list[Point]) -> None:
        mesh_edges = {
            self._edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }
        for idx in range(len(hole)):
            self.assertIn(self._edge_key((hole[idx], hole[(idx + 1) % len(hole)])), mesh_edges)

    def _same_undirected_edge(self, first: tuple[Point, Point], second: tuple[Point, Point]) -> bool:
        return self._edge_key(first) == self._edge_key(second)

    def _edge_key(self, edge: tuple[Point, Point]) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = edge
        return tuple(sorted(((round(a.x, 6), round(a.y, 6)), (round(b.x, 6), round(b.y, 6)))))

    def _mesh_respects_cut_line(self, mesh: Mesh, cut: list[Point]) -> bool:
        if len(cut) < 2:
            return True
        line = (cut[0], cut[-1])
        for triangle in mesh.triangles:
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)):
                if segments_intersect(edge[0], edge[1], line[0], line[1], include_endpoints=False):
                    return False
        return True


if __name__ == "__main__":
    unittest.main()
