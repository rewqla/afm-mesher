import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import src.application.services.mesh_postprocessing as mesh_postprocessing
from src.application.services.mesh_postprocessing import renumber_triangles
from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point


class TestTriangleNumbering(unittest.TestCase):
    def test_basic_numbering_reassigns_dense_range(self) -> None:
        triangles = [
            self._triangle(30, (3, 5, 7)),
            self._triangle(10, (1, 4, 5)),
            self._triangle(20, (2, 3, 6)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertEqual([triangle.triangle_number for triangle in renumbered], [1, 2, 3])

    def test_sorts_by_min_node_number(self) -> None:
        triangles = [
            self._triangle(90, (3, 5, 7)),
            self._triangle(80, (2, 3, 6)),
            self._triangle(70, (1, 4, 5)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertEqual(
            [(triangle.triangle_number, triangle.node_numbers) for triangle in renumbered],
            [(1, (1, 4, 5)), (2, (2, 3, 6)), (3, (3, 5, 7))],
        )

    def test_resolves_same_min_by_max_node_number(self) -> None:
        triangles = [
            self._triangle(20, (1, 3, 6)),
            self._triangle(10, (1, 2, 4)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertEqual(renumbered[0].node_numbers, (1, 2, 4))
        self.assertEqual(renumbered[0].triangle_number, 1)
        self.assertEqual(renumbered[1].node_numbers, (1, 3, 6))
        self.assertEqual(renumbered[1].triangle_number, 2)

    def test_preserves_ccw_orientation(self) -> None:
        triangles = [
            self._triangle(5, (3, 5, 7)),
            self._triangle(4, (1, 4, 5)),
            self._triangle(3, (2, 3, 6)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertTrue(all(triangle.area > 0.0 for triangle in renumbered))
        self.assertTrue(all(triangle.orientation() == 1 for triangle in renumbered))

    def test_preserves_node_numbers(self) -> None:
        triangles = [
            self._triangle(9, (3, 5, 7)),
            self._triangle(8, (1, 4, 5)),
            self._triangle(7, (2, 3, 6)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertEqual(
            {triangle.node_numbers for triangle in renumbered},
            {triangle.node_numbers for triangle in triangles},
        )

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(renumber_triangles([]), [])

    def test_single_triangle_is_renumbered_to_one(self) -> None:
        triangle = self._triangle(99, (5, 8, 13))

        renumbered = renumber_triangles([triangle])

        self.assertEqual(len(renumbered), 1)
        self.assertEqual(renumbered[0].triangle_number, 1)
        self.assertEqual(renumbered[0].node_numbers, (5, 8, 13))

    def test_stable_sort_preserves_input_order_for_identical_keys(self) -> None:
        first = self._triangle(20, (1, 2, 5))
        second = self._triangle(10, (1, 3, 5))

        renumbered = renumber_triangles([first, second])

        self.assertEqual(
            [triangle.node_numbers for triangle in renumbered],
            [(1, 2, 5), (1, 3, 5)],
        )
        self.assertEqual([triangle.triangle_number for triangle in renumbered], [1, 2])

    def test_raises_import_error_when_numpy_is_unavailable(self) -> None:
        with patch.object(mesh_postprocessing, "np", None):
            with self.assertRaisesRegex(ImportError, "numpy is required for renumber_triangles"):
                renumber_triangles([self._triangle(1, (1, 2, 3))])

    def test_numbering_is_global_for_entire_mesh(self) -> None:
        # Simulates triangles coming from main domain and inclusion-related geometry
        # after they were already merged into one mesh-wide list.
        triangles = [
            self._triangle(100, (4, 8, 9)),
            self._triangle(200, (1, 5, 6)),
            self._triangle(300, (2, 6, 7)),
            self._triangle(400, (10, 11, 12)),
        ]

        renumbered = renumber_triangles(triangles)

        self.assertEqual([triangle.triangle_number for triangle in renumbered], [1, 2, 3, 4])
        self.assertEqual(
            [triangle.node_numbers for triangle in renumbered],
            [(1, 5, 6), (2, 6, 7), (4, 8, 9), (10, 11, 12)],
        )

    def test_large_input_produces_dense_unique_numbers(self) -> None:
        triangles = [
            self._triangle(
                triangle_number=50_000 - index,
                node_numbers=(index + 1, index + 2, index + 3),
            )
            for index in reversed(range(50_000))
        ]

        renumbered = renumber_triangles(triangles)

        numbers = [triangle.triangle_number for triangle in renumbered]
        self.assertEqual(len(numbers), 50_000)
        self.assertEqual(numbers[0], 1)
        self.assertEqual(numbers[-1], 50_000)
        self.assertEqual(set(numbers), set(range(1, 50_001)))

    def _triangle(
        self,
        triangle_number: int,
        node_numbers: tuple[int, int, int],
    ) -> LinearTriangle:
        a_id, b_id, c_id = node_numbers
        return LinearTriangle(
            triangle_number=triangle_number,
            node_coordinates=(
                Point(float(a_id), 0.0),
                Point(float(b_id), 0.0),
                Point(float(c_id), 1.0),
            ),
            node_numbers=node_numbers,
        )


if __name__ == "__main__":
    unittest.main()
