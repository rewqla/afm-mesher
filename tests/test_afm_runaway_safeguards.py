import unittest
from time import monotonic
from unittest.mock import patch

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import (
    AdvancingFrontMesher,
    MeshingTimeoutError,
    _TriangleSpatialIndex,
)
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter

SQUARE = [Point(0.0, 0.0), Point(80.0, 0.0), Point(80.0, 80.0), Point(0.0, 80.0)]


class TestAfmRunawaySafeguards(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(min_triangle_quality=0.01, target_edge_length=20.0, smoothing_iterations=0)

    def test_expired_deadline_raises_timeout_instead_of_meshing(self) -> None:
        with self.assertRaises(MeshingTimeoutError):
            self.mesher.generate_with_holes_and_cuts(SQUARE, holes=[], cuts=[], deadline=monotonic() - 1.0)

    def test_time_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            AdvancingFrontMesher(time_budget_s=0.0)

    def test_adapter_does_not_retry_cut_variants_after_timeout(self) -> None:
        adapter = TriangulationAdapter()
        outer = [(0, 0), (160, 0), (160, 130), (0, 130), (0, 0)]
        cut = [(40, 0), (40, 80)]
        with patch(
            "src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts",
            side_effect=MeshingTimeoutError("timeout"),
        ) as generate:
            with self.assertRaises(MeshingTimeoutError):
                adapter.run_from_contours([outer, cut], mode=TriangulationMode.FAST)
        self.assertEqual(generate.call_count, 1)

    def test_candidate_overlapping_built_triangle_is_rejected(self) -> None:
        index = _TriangleSpatialIndex(cell_size=15.0)
        index.insert(Point(0.0, 0.0), Point(10.0, 0.0), Point(0.0, 10.0))
        self.mesher._built_index = index
        # Same base edge, apex on the already meshed side.
        self.assertTrue(self.mesher._overlaps_built_triangles(Point(0.0, 0.0), Point(10.0, 0.0), Point(2.0, 2.0)))
        # Opposite side of the shared edge is fine.
        self.assertFalse(self.mesher._overlaps_built_triangles(Point(10.0, 0.0), Point(0.0, 0.0), Point(5.0, -8.0)))
        # Apex on the side of the built triangle is a hanging node.
        self.assertTrue(self.mesher._overlaps_built_triangles(Point(10.0, 0.0), Point(10.0, 10.0), Point(5.0, 5.0)))

    def test_final_validation_rejects_overlap_and_hanging_node(self) -> None:
        a, b, c, d = Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0)
        valid = Mesh(triangles=[Triangle(a, b, c), Triangle(a, c, d)])
        self.mesher._validate_final_mesh(valid, domain_area=100.0)

        with self.assertRaises(ValueError):
            self.mesher._validate_final_mesh(valid, domain_area=90.0)

        overlapping = Mesh(triangles=[Triangle(a, b, c), Triangle(a, b, d)])
        with self.assertRaises(ValueError):
            self.mesher._validate_final_mesh(overlapping, domain_area=100.0)

        mid = Point(5.0, 5.0)
        hanging = Mesh(triangles=[Triangle(a, b, c), Triangle(a, mid, d), Triangle(mid, c, d)])
        with self.assertRaises(ValueError):
            self.mesher._validate_final_mesh(hanging, domain_area=100.0)

    def test_cut_endpoint_inside_boundary_edge_becomes_mesh_vertex(self) -> None:
        cut = [Point(37.0, 0.0), Point(37.0, 50.0)]
        mesh = self.mesher.generate_with_holes_and_cuts(SQUARE, holes=[], cuts=[cut])
        vertices = {(round(p.x, 6), round(p.y, 6)) for t in mesh.triangles for p in (t.a, t.b, t.c)}
        self.assertIn((37.0, 0.0), vertices)
        self.assertAlmostEqual(sum(t.area() for t in mesh.triangles), 6400.0, places=6)


if __name__ == "__main__":
    unittest.main()
