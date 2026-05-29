import unittest

import _bootstrap  # noqa: F401
from src.application.services.boundary_validator import BoundaryValidator


class TestBoundaryValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = BoundaryValidator(closure_tolerance=2.0, epsilon=0.5)

    def test_open_contour_detected(self) -> None:
        contours = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
        result = self.validator.validate(contours)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def test_two_point_open_contour_is_reported_as_open(self) -> None:
        contours = [[(4, 3), (4, 4)]]
        result = self.validator.validate(contours)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def test_closed_loop_with_tail_is_normalized_and_valid(self) -> None:
        contour = [[(0, 0), (10, 0), (10, 10), (0, 10), (0.4, 0.4), (2, -2), (3, -3)]]
        result = self.validator.validate(contour)
        normalized = self.validator.normalize_closed_contours(contour)

        self.assertTrue(result.is_valid)
        self.assertEqual(normalized[0][0], normalized[0][-1])
        self.assertEqual(len(normalized[0]), 5)

    def test_self_intersection_detected(self) -> None:
        contours = [[(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)]]
        result = self.validator.validate(contours)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.self_intersections), 1)

    def test_normalization_keeps_open_contour_open(self) -> None:
        contour = [[(0, 0), (10, 0), (10, 10), (1, 10)]]
        normalized = self.validator.normalize_closed_contours(contour)

        self.assertNotEqual(normalized[0][0], normalized[0][-1])

    def test_closed_loop_with_tail_after_return_to_start_is_valid(self) -> None:
        contour = [[(0, 0), (10, 0), (10, 10), (0, 10), (1, 1), (3, 2)]]
        result = self.validator.validate(contour)
        normalized = self.validator.normalize_closed_contours(contour)

        self.assertTrue(result.is_valid)
        self.assertEqual(normalized[0][0], normalized[0][-1])

    def test_select_outer_contour_ignores_inner_open_cut(self) -> None:
        outer = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        cut = [(5, 5), (15, 15)]

        selected = self.validator.select_outer_contour([outer, cut])
        closed = self.validator.normalize_valid_closed_contours([outer, cut])

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected[0], selected[-1])
        self.assertEqual(len(closed), 1)

    def test_validate_checks_only_outer_contour_when_closed_shell_exists(self) -> None:
        outer = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
        inner_open = [(8, 8), (12, 20)]

        result = self.validator.validate([outer, inner_open])

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.open_contours), 0)
        self.assertEqual(len(result.self_intersections), 0)


if __name__ == "__main__":
    unittest.main()
