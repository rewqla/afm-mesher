import unittest

import _bootstrap  # noqa: F401

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.application.services.mesh_postprocessing import laplacian_smooth
from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.node_type import NodeType
from src.domain.entities.point import Point


class TestNodeClassification(unittest.TestCase):
    def test_single_hole_classifies_boundary_interface_and_interior_nodes(self) -> None:
        mesher = AdvancingFrontMesher(min_triangle_quality=0.0, target_edge_length=10.0, smoothing_iterations=0)
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
        indexed = mesh.indexed_mesh_data

        self.assertIsNotNone(indexed)
        assert indexed is not None
        self.assertEqual(indexed.boundary_nodes & indexed.interface_nodes, set())

        boundary_coords = self._coords_for_ids(indexed, indexed.boundary_nodes)
        interface_coords = self._coords_for_ids(indexed, indexed.interface_nodes)

        self.assertIn((0.0, 0.0), boundary_coords)
        self.assertIn((100.0, 100.0), boundary_coords)
        self.assertIn((40.0, 40.0), interface_coords)
        self.assertIn((60.0, 60.0), interface_coords)

        interior_ids = {
            node_id
            for node_id in range(indexed.index_base, indexed.index_base + len(indexed.nodes))
            if indexed.node_type(node_id) == NodeType.INTERIOR
        }
        self.assertTrue(interior_ids)

    def test_area_without_inclusions_keeps_interface_nodes_empty(self) -> None:
        mesher = AdvancingFrontMesher(min_triangle_quality=0.0, target_edge_length=12.0, smoothing_iterations=0)
        boundary = [
            Point(0.0, 0.0),
            Point(80.0, 0.0),
            Point(80.0, 80.0),
            Point(0.0, 80.0),
        ]

        mesh = mesher.generate(boundary)
        indexed = mesh.indexed_mesh_data

        self.assertIsNotNone(indexed)
        assert indexed is not None
        self.assertEqual(indexed.interface_nodes, set())
        self.assertTrue(indexed.boundary_nodes)

        boundary_coords = self._coords_for_ids(indexed, indexed.boundary_nodes)
        for point in boundary:
            self.assertIn((point.x, point.y), boundary_coords)

    def test_multiple_holes_collect_all_interface_nodes(self) -> None:
        mesher = AdvancingFrontMesher(min_triangle_quality=0.0, target_edge_length=10.0, smoothing_iterations=0)
        boundary = [
            Point(0.0, 0.0),
            Point(140.0, 0.0),
            Point(140.0, 120.0),
            Point(0.0, 120.0),
        ]
        holes = [
            [Point(20.0, 20.0), Point(40.0, 20.0), Point(40.0, 40.0), Point(20.0, 40.0)],
            [Point(90.0, 20.0), Point(115.0, 20.0), Point(115.0, 45.0), Point(90.0, 45.0)],
        ]

        mesh = mesher.generate_with_holes(boundary, holes)
        indexed = mesh.indexed_mesh_data

        self.assertIsNotNone(indexed)
        assert indexed is not None
        interface_coords = self._coords_for_ids(indexed, indexed.interface_nodes)

        self.assertIn((20.0, 20.0), interface_coords)
        self.assertIn((40.0, 40.0), interface_coords)
        self.assertIn((90.0, 20.0), interface_coords)
        self.assertIn((115.0, 45.0), interface_coords)

    def test_laplacian_smooth_keeps_boundary_and_interface_nodes_fixed(self) -> None:
        nodes = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (2.0, 1.0),
            (2.8, 2.2),
        ]
        triangles = [
            (0, 1, 4),
            (1, 2, 5),
            (2, 3, 5),
            (3, 0, 4),
            (0, 4, 5),
            (0, 5, 3),
            (1, 5, 4),
        ]

        smoothed = laplacian_smooth(
            nodes,
            triangles,
            boundary_nodes={0, 1, 2, 3},
            interface_nodes={4},
            iterations=5,
        )

        self.assertEqual(smoothed[0], nodes[0])
        self.assertEqual(smoothed[1], nodes[1])
        self.assertEqual(smoothed[2], nodes[2])
        self.assertEqual(smoothed[3], nodes[3])
        self.assertEqual(smoothed[4], nodes[4])
        self.assertNotEqual(smoothed[5], nodes[5])

    def test_node_type_prioritizes_interface_over_boundary(self) -> None:
        indexed = IndexedMesh(
            nodes=[(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)],
            triangles=[(1, 2, 3)],
            boundary_nodes={1, 2},
            interface_nodes={2},
            bandwidth=2,
            index_base=1,
        )

        self.assertEqual(indexed.node_type(1), NodeType.BOUNDARY)
        self.assertEqual(indexed.node_type(2), NodeType.INTERFACE)
        self.assertEqual(indexed.node_type(3), NodeType.INTERIOR)

    def _coords_for_ids(self, indexed: IndexedMesh, node_ids: set[int]) -> set[tuple[float, float]]:
        return {indexed.nodes[node_id - indexed.index_base] for node_id in node_ids}


if __name__ == "__main__":
    unittest.main()
