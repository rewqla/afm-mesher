import unittest

import _bootstrap  # noqa: F401
from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle


class TestLinearTriangle(unittest.TestCase):
    def _valid_triangle(self) -> LinearTriangle:
        return LinearTriangle(
            triangle_number=10,
            node_coordinates=(Point(0.0, 0.0), Point(4.0, 0.0), Point(0.0, 2.0)),
            node_numbers=(101, 102, 103),
        )

    def test_01_create_valid_linear_triangle(self) -> None:
        triangle = self._valid_triangle()
        self.assertIsInstance(triangle, LinearTriangle)
        self.assertIsInstance(triangle, Triangle)

    def test_02_preserves_triangle_id(self) -> None:
        triangle = self._valid_triangle()
        self.assertEqual(triangle.triangle_id, 10)
        self.assertEqual(triangle.triangle_number, 10)

    def test_03_preserves_node_ids(self) -> None:
        triangle = self._valid_triangle()
        self.assertEqual(triangle.node_ids, (101, 102, 103))
        self.assertEqual(triangle.node_numbers, (101, 102, 103))

    def test_04_preserves_node_coordinates(self) -> None:
        triangle = self._valid_triangle()
        self.assertEqual(
            triangle.node_coordinates,
            (Point(0.0, 0.0), Point(4.0, 0.0), Point(0.0, 2.0)),
        )

    def test_05_computes_area_correctly(self) -> None:
        triangle = self._valid_triangle()
        self.assertAlmostEqual(triangle.area, 4.0, places=12)

    def test_06_orientation_is_positive(self) -> None:
        triangle = self._valid_triangle()
        signed = LinearTriangle.compute_area(triangle.node_coordinates, signed=True)
        self.assertGreater(signed, 0.0)

    def test_07_negative_orientation_is_auto_reordered(self) -> None:
        triangle = LinearTriangle(
            triangle_number=1,
            node_coordinates=(Point(0.0, 0.0), Point(0.0, 2.0), Point(4.0, 0.0)),
            node_numbers=(11, 22, 33),
        )
        self.assertEqual(triangle.node_numbers, (11, 33, 22))
        signed = LinearTriangle.compute_area(triangle.node_coordinates, signed=True)
        self.assertGreater(signed, 0.0)

    def test_08_rejects_degenerate_triangle(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)),
                node_numbers=(1, 2, 3),
            )

    def test_09_rejects_not_three_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0)),
                node_numbers=(1, 2, 3),
            )
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0), Point(2.0, 2.0)),
                node_numbers=(1, 2, 3),
            )

    def test_10_rejects_not_three_node_ids(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(1, 2),
            )
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(1, 2, 3, 4),
            )

    def test_11_rejects_non_positive_node_id(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(0, 2, 3),
            )
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(1, -2, 3),
            )

    def test_12_rejects_non_positive_triangle_id(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=0,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(1, 2, 3),
            )
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=-1,
                node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                node_numbers=(1, 2, 3),
            )

    def test_13_rejects_repeated_node_ids(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(2.0, 0.0), Point(0.0, 2.0)),
                node_numbers=(5, 5, 6),
            )

    def test_14_rejects_repeated_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            LinearTriangle(
                triangle_number=1,
                node_coordinates=(Point(0.0, 0.0), Point(0.0, 0.0), Point(1.0, 1.0)),
                node_numbers=(1, 2, 3),
            )

    def test_15_phi_i_in_vertices(self) -> None:
        t = self._valid_triangle()
        i, j, k = t.node_coordinates
        self.assertAlmostEqual(t.phi_i(i.x, i.y), 1.0, places=12)
        self.assertAlmostEqual(t.phi_i(j.x, j.y), 0.0, places=12)
        self.assertAlmostEqual(t.phi_i(k.x, k.y), 0.0, places=12)

    def test_16_phi_j_in_vertices(self) -> None:
        t = self._valid_triangle()
        i, j, k = t.node_coordinates
        self.assertAlmostEqual(t.phi_j(i.x, i.y), 0.0, places=12)
        self.assertAlmostEqual(t.phi_j(j.x, j.y), 1.0, places=12)
        self.assertAlmostEqual(t.phi_j(k.x, k.y), 0.0, places=12)

    def test_17_phi_k_in_vertices(self) -> None:
        t = self._valid_triangle()
        i, j, k = t.node_coordinates
        self.assertAlmostEqual(t.phi_k(i.x, i.y), 0.0, places=12)
        self.assertAlmostEqual(t.phi_k(j.x, j.y), 0.0, places=12)
        self.assertAlmostEqual(t.phi_k(k.x, k.y), 1.0, places=12)

    def test_18_phi_sum_is_one_in_inner_point(self) -> None:
        t = self._valid_triangle()
        x, y = 1.0, 0.5
        value = t.phi_i(x, y) + t.phi_j(x, y) + t.phi_k(x, y)
        self.assertAlmostEqual(value, 1.0, places=12)

    def test_19_same_node_id_can_be_in_different_triangles(self) -> None:
        t1 = LinearTriangle(
            triangle_number=1,
            node_coordinates=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
            node_numbers=(7, 8, 9),
        )
        t2 = LinearTriangle(
            triangle_number=2,
            node_coordinates=(Point(1.0, 0.0), Point(2.0, 0.0), Point(1.0, 1.0)),
            node_numbers=(8, 10, 11),
        )
        self.assertIn(8, t1.node_numbers)
        self.assertIn(8, t2.node_numbers)

    def test_20_no_assumption_nodes_count_equals_triangles_count(self) -> None:
        t = LinearTriangle(
            triangle_number=1,
            node_coordinates=(Point(0.0, 0.0), Point(3.0, 0.0), Point(0.0, 1.0)),
            node_numbers=(100, 200, 300),
        )
        self.assertNotEqual(t.triangle_number, len(t.node_numbers))
        self.assertNotEqual(t.triangle_number, t.node_numbers[0])


if __name__ == "__main__":
    unittest.main()
