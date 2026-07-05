import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
import src.application.services.advancing_front_mesher as afm_module
from src.application.services.mesh_postprocessing import (
    _build_node_adjacency,
    _compute_bandwidth,
    _normalize_triangle_indices,
    _select_sloan_seed_nodes,
    improve_mesh_by_edge_flips,
    laplacian_smooth,
    renumber_nodes_multilevel_rcm,
    renumber_nodes_gps,
    renumber_nodes_rcm,
    renumber_nodes_rcm_multistart,
    refine_numbering_local_search,
    renumber_nodes_sloan_multistart,
)
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary
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

    def test_renumber_nodes_rcm_multistart_is_not_worse_than_single_start(self) -> None:
        # Multi-start should never be worse than the baseline single-start RCM on the same graph.
        nodes = [
            (1.0, 1.0),
            (0.0, 0.0),
            (2.0, 0.0),
            (3.0, 1.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (-1.0, 1.0),
        ]
        triangles = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 5),
            (0, 5, 6),
            (0, 6, 1),
        ]

        _, _, single_beta = renumber_nodes_rcm(nodes, triangles)
        _, _, multistart_beta = renumber_nodes_rcm_multistart(nodes, triangles)

        self.assertLessEqual(multistart_beta, single_beta)

    def test_renumber_nodes_rcm_multistart_checkpoint_betas(self) -> None:
        cases = [
            ("one.png", 15),
            ("spot_with_circle.png", 37),
            ("spot_with_lines.png", 37),
        ]
        for image_name, expected_beta in cases:
            with self.subTest(image=image_name):
                nodes, triangles = _capture_raw_rcm_graph(image_name)
                _, _, beta = renumber_nodes_rcm_multistart(nodes, triangles)
                self.assertEqual(beta, expected_beta)

    def test_renumber_nodes_multilevel_rcm_returns_valid_permutation_for_checkpoint_graphs(self) -> None:
        cases = [
            ("one.png", 50),
            ("spot_with_circle.png", 75),
            ("spot_with_lines.png", 150),
        ]
        for image_name, leaf_size in cases:
            with self.subTest(image=image_name, leaf_size=leaf_size):
                nodes, triangles = _capture_raw_rcm_graph(image_name)
                new_nodes, new_triangles, beta = renumber_nodes_multilevel_rcm(nodes, triangles, leaf_size=leaf_size)

                self.assertEqual(len(new_nodes), len(nodes))
                self.assertEqual(sorted(new_nodes), sorted(nodes))
                triangle_node_ids = {node_id for tri in new_triangles for node_id in tri}
                expected_zero_based = set(range(len(nodes)))
                expected_one_based = set(range(1, len(nodes) + 1))
                self.assertIn(triangle_node_ids, (expected_zero_based, expected_one_based))
                self.assertEqual(beta, _compute_bandwidth(new_triangles))

    def test_renumber_nodes_multilevel_rcm_rejects_invalid_leaf_size(self) -> None:
        with self.assertRaises(ValueError):
            renumber_nodes_multilevel_rcm([(0.0, 0.0)], [(0, 0, 0)], leaf_size=0)

    def test_renumber_nodes_gps_returns_valid_permutation(self) -> None:
        nodes = [
            (1.0, 1.0),
            (0.0, 0.0),
            (2.0, 0.0),
            (3.0, 1.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (-1.0, 1.0),
        ]
        triangles = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 5),
            (0, 5, 6),
            (0, 6, 1),
        ]

        new_nodes, new_triangles, beta = renumber_nodes_gps(nodes, triangles)

        self.assertEqual(len(new_nodes), len(nodes))
        self.assertEqual(sorted(new_nodes), sorted(nodes))
        self.assertEqual(set(node_id for tri in new_triangles for node_id in tri), set(range(len(nodes))))
        self.assertEqual(beta, max(max(triangle) - min(triangle) for triangle in new_triangles))

    def test_renumber_nodes_gps_checkpoint_betas_against_rcm_multistart(self) -> None:
        cases = [
            ("one.png", 15, 22),
            ("spot_with_circle.png", 37, 72),
            ("spot_with_lines.png", 37, 72),
        ]
        for image_name, expected_rcm_beta, expected_gps_beta in cases:
            with self.subTest(image=image_name):
                nodes, triangles = _capture_raw_rcm_graph(image_name)
                _, _, rcm_beta = renumber_nodes_rcm_multistart(nodes, triangles)
                _, _, gps_beta = renumber_nodes_gps(nodes, triangles)

                self.assertEqual(rcm_beta, expected_rcm_beta)
                self.assertEqual(gps_beta, expected_gps_beta)
                self.assertGreaterEqual(gps_beta, rcm_beta)

    def test_refine_numbering_local_search_is_never_worse_than_rcm_multistart(self) -> None:
        cases = [
            ("one.png", 15),
            ("spot_with_circle.png", 37),
            ("spot_with_lines.png", 37),
        ]
        for image_name, expected_beta in cases:
            with self.subTest(image=image_name):
                nodes, triangles = _capture_raw_rcm_graph(image_name)
                baseline_nodes, baseline_triangles, baseline_beta = renumber_nodes_rcm_multistart(nodes, triangles)
                refined_nodes, refined_triangles, refined_beta = refine_numbering_local_search(
                    baseline_nodes,
                    baseline_triangles,
                )
                self.assertEqual(baseline_beta, expected_beta)
                self.assertLessEqual(refined_beta, baseline_beta)
                self.assertEqual(refined_beta, _compute_bandwidth(refined_triangles))
                self.assertEqual(len(refined_nodes), len(baseline_nodes))

    def test_select_sloan_seed_nodes_includes_probe_endpoints_for_one_image(self) -> None:
        nodes, triangles = _capture_raw_sloan_graph("one.png")
        normalized_triangles, _ = _normalize_triangle_indices(triangles, len(nodes))
        adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
        degrees = [len(neighbors) for neighbors in adjacency]
        seeds = _select_sloan_seed_nodes(adjacency, degrees, max_candidates=5)

        self.assertTrue(any(degrees[node_id] > min(degrees) for node_id in seeds))

    def test_renumber_nodes_sloan_multistart_keeps_one_image_bandwidth_reasonable(self) -> None:
        nodes, triangles = _capture_raw_sloan_graph("one.png")
        _, _, beta = renumber_nodes_sloan_multistart(nodes, triangles)

        self.assertLessEqual(beta, 37)

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


def _capture_raw_sloan_graph(image_name: str) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    return _capture_raw_graph(image_name, "sloan_multistart")


def _capture_raw_rcm_graph(image_name: str) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    return _capture_raw_graph(image_name, "rcm_multistart")


def _capture_raw_graph(
    image_name: str,
    numbering_strategy: str,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "data" / "images" / image_name
    boundary = preprocess_photo_to_boundary(image_path, threshold=127, epsilon=2.0).boundary
    mesher = AdvancingFrontMesher(
        min_triangle_quality=0.01,
        max_iterations_factor=180,
        target_edge_length=24.0,
        smoothing_iterations=2,
        numbering_strategy=numbering_strategy,
    )

    captured: dict[str, list[tuple[float, float]] | list[tuple[int, int, int]]] = {}
    if numbering_strategy == "rcm_multistart":
        original = afm_module.renumber_nodes_rcm_multistart

        def _stub(
            nodes: list[tuple[float, float]],
            triangles: list[tuple[int, int, int]],
            candidates: list[int] | None = None,
            max_candidates: int = 15,
            probe_stride: int = 12,
        ) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
            captured["nodes"] = list(nodes)
            captured["triangles"] = [tuple(triangle) for triangle in triangles]
            return list(nodes), [tuple(triangle) for triangle in triangles], _compute_bandwidth([tuple(triangle) for triangle in triangles])
    elif numbering_strategy == "sloan_multistart":
        original = afm_module.renumber_nodes_sloan_multistart

        def _stub(
            nodes: list[tuple[float, float]],
            triangles: list[tuple[int, int, int]],
            max_candidates: int = 5,
        ) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
            captured["nodes"] = list(nodes)
            captured["triangles"] = [tuple(triangle) for triangle in triangles]
            return list(nodes), [tuple(triangle) for triangle in triangles], _compute_bandwidth([tuple(triangle) for triangle in triangles])
    else:
        raise AssertionError(f"Unsupported numbering strategy for capture: {numbering_strategy}")

    if numbering_strategy == "rcm_multistart":
        afm_module.renumber_nodes_rcm_multistart = _stub  # type: ignore[assignment]
    else:
        afm_module.renumber_nodes_sloan_multistart = _stub  # type: ignore[assignment]
    try:
        mesher.generate(boundary)
    finally:
        if numbering_strategy == "rcm_multistart":
            afm_module.renumber_nodes_rcm_multistart = original  # type: ignore[assignment]
        else:
            afm_module.renumber_nodes_sloan_multistart = original  # type: ignore[assignment]

    nodes = captured.get("nodes")
    triangles = captured.get("triangles")
    if not isinstance(nodes, list) or not isinstance(triangles, list):
        raise AssertionError("Failed to capture raw Sloan graph data.")
    return [tuple(node) for node in nodes], [tuple(triangle) for triangle in triangles]


if __name__ == "__main__":
    unittest.main()
