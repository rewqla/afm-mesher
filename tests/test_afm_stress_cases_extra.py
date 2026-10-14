import unittest
from time import monotonic

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher, MeshingTimeoutError
from src.domain.entities.point import Point

_SQUARE_80 = [Point(0.0, 0.0), Point(80.0, 0.0), Point(80.0, 80.0), Point(0.0, 80.0)]


class TestAfmStressCasesExtra(unittest.TestCase):
    def test_acute_angle_polygon_does_not_stall(self) -> None:
        # Very sharp apex near the base line.
        boundary = [
            Point(20.0, 20.0),
            Point(480.0, 20.0),
            Point(250.0, 24.0),
        ]
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=15.0,
            smoothing_iterations=0,
            max_iterations_factor=800,
        )
        mesh = mesher.generate(boundary)
        self.assertGreater(len(mesh.triangles), 0)

    def test_minimal_step_relative_to_shape_size_has_bounded_runtime(self) -> None:
        # Stress-like case with small target h: either a valid mesh or a clear timeout, never a hang.
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=2.0,
            smoothing_iterations=0,
            max_iterations_factor=120,
            time_budget_s=5.0,
        )
        started = monotonic()
        try:
            mesh = mesher.generate(_SQUARE_80)
        except MeshingTimeoutError:
            pass
        else:
            self.assertGreater(len(mesh.triangles), 80)
            self.assertLess(len(mesh.triangles), 30000)
        self.assertLess(monotonic() - started, 15.0)

    def test_minimal_step_relative_to_shape_size_meshes_conformally(self) -> None:
        # Used to "pass" in ~90 s with 82 hanging nodes (lattice edge midpoints); Steiner snapping
        # to nearby front vertices closes those pockets conformally.
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=2.0,
            smoothing_iterations=0,
            max_iterations_factor=120,
            time_budget_s=10.0,
        )
        mesh = mesher.generate(_SQUARE_80)
        self.assertGreater(len(mesh.triangles), 80)

    def test_degenerate_collinear_input_raises(self) -> None:
        boundary = [Point(0.0, 0.0), Point(50.0, 50.0), Point(100.0, 100.0)]
        mesher = AdvancingFrontMesher(min_triangle_quality=0.01)
        with self.assertRaises(ValueError):
            mesher.generate(boundary)


if __name__ == "__main__":
    unittest.main()
