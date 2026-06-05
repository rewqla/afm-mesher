import unittest

import _bootstrap  # noqa: F401
from src.application.services.mesh_postprocessing import (
    improve_mesh_by_edge_flips,
    laplacian_smooth,
    renumber_nodes_rcm,
)
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import orientation, triangle_quality


class TestMeshPostprocessing(unittest.TestCase):
    def test_renumber_nodes_rcm_reduces_bandwidth_for_zero_based_indices(self) -> None:
        nodes = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]
        triangles = [
            (0, 2, 4),
            (0, 4, 3),
            (1, 5, 4),
            (1, 4, 2),
        ]

        new_nodes, new_triangles, bandwidth = renumber_nodes_rcm(nodes, triangles)

        self.assertEqual(len(new_nodes), len(nodes))
        self.assertEqual(sorted(new_nodes), sorted(nodes))
        self.assertEqual(set(node_id for tri in new_triangles for node_id in tri), set(range(len(nodes))))
        original_bandwidth = max(max(triangle) - min(triangle) for triangle in triangles)
        self.assertLessEqual(bandwidth, original_bandwidth)
        self.assertEqual(bandwidth, max(max(triangle) - min(triangle) for triangle in new_triangles))

    def test_renumber_nodes_rcm_preserves_one_based_indexing(self) -> None:
        nodes = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        triangles = [
            (1, 2, 3),
            (1, 3, 4),
        ]

        new_nodes, new_triangles, bandwidth = renumber_nodes_rcm(nodes, triangles)

        self.assertEqual(sorted(new_nodes), sorted(nodes))
        self.assertEqual(set(node_id for tri in new_triangles for node_id in tri), {1, 2, 3, 4})
        self.assertTrue(all(min(triangle) >= 1 for triangle in new_triangles))
        self.assertEqual(bandwidth, max(max(triangle) - min(triangle) for triangle in new_triangles))

    def test_laplacian_smooth_keeps_boundary_fixed_and_improves_quality(self) -> None:
        nodes = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (80.0, 50.0),
        ]
        triangles = [
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ]
        boundary_nodes = {0, 1, 2, 3}

        before_quality = _mean_quality(nodes, triangles)
        smoothed = laplacian_smooth(nodes, triangles, boundary_nodes=boundary_nodes, iterations=5)
        after_quality = _mean_quality(smoothed, triangles)

        self.assertGreaterEqual(after_quality, before_quality)
        self.assertEqual(smoothed[:4], nodes[:4])
        self.assertLess(abs(smoothed[4][0] - 50.0), abs(nodes[4][0] - 50.0))
        self.assertLessEqual(abs(smoothed[4][1] - 50.0), abs(nodes[4][1] - 50.0))

    def test_laplacian_smooth_rolls_back_move_that_inverts_triangle(self) -> None:
        nodes = [
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
            (1.0, 0.0),
        ]
        triangles = [
            (0, 1, 2),
            (3, 1, 2),
            (0, 3, 2),
        ]
        boundary_nodes = {0, 1, 3}

        smoothed = laplacian_smooth(nodes, triangles, boundary_nodes=boundary_nodes, iterations=1)

        self.assertEqual(smoothed[2], nodes[2])
        for a_id, b_id, c_id in triangles:
            a = Point(*smoothed[a_id])
            b = Point(*smoothed[b_id])
            c = Point(*smoothed[c_id])
            self.assertGreater(orientation(a, b, c), 0)

    def test_edge_flip_improves_local_diagonal_quality(self) -> None:
        nodes = [
            (0.0, 0.0),
            (3.0, 0.0),
            (1.5, 0.2),
            (0.0, 2.0),
        ]
        triangles = [
            (0, 1, 2),
            (0, 2, 3),
        ]

        before_quality = _mean_quality(nodes, triangles)
        flipped = improve_mesh_by_edge_flips(nodes, triangles, max_passes=2)
        after_quality = _mean_quality(nodes, flipped)

        self.assertGreater(after_quality, before_quality)
        self.assertEqual({tuple(sorted(tri)) for tri in flipped}, {tuple(sorted((0, 1, 3))), tuple(sorted((1, 2, 3)))})


def _mean_quality(nodes: list[tuple[float, float]], triangles: list[tuple[int, int, int]]) -> float:
    qualities: list[float] = []
    for a_id, b_id, c_id in triangles:
        a = Point(*nodes[a_id])
        b = Point(*nodes[b_id])
        c = Point(*nodes[c_id])
        qualities.append(triangle_quality(a, b, c))
    return sum(qualities) / len(qualities)


if __name__ == "__main__":
    unittest.main()
