import unittest
from math import hypot

import _bootstrap  # noqa: F401
from src.application.dto.complex_test_contours import (
    hourglass_contour,
    star_contour,
    u_shape_contour,
)


class TestComplexTestContours(unittest.TestCase):
    def test_star_has_10_points_and_in_bounds(self) -> None:
        points = star_contour()
        self.assertEqual(len(points), 10)
        self._assert_in_bounds(points)
        self._assert_star_alternating_radii(points, center_x=250.0, center_y=250.0, outer=200.0, inner=80.0)

    def test_u_shape_in_bounds(self) -> None:
        points = u_shape_contour()
        self.assertGreaterEqual(len(points), 8)
        self._assert_in_bounds(points)

    def test_hourglass_in_bounds(self) -> None:
        points = hourglass_contour()
        self.assertGreaterEqual(len(points), 8)
        self._assert_in_bounds(points)

    def test_u_shape_is_concave(self) -> None:
        points = u_shape_contour()
        self.assertTrue(self._is_concave(points))

    def test_hourglass_has_narrow_neck(self) -> None:
        points = hourglass_contour()
        xs = sorted({p.x for p in points})
        # Hourglass corridor boundaries are expected around x=220 and x=280.
        self.assertIn(220.0, xs)
        self.assertIn(280.0, xs)
        self.assertAlmostEqual(280.0 - 220.0, 60.0, places=6)

    def _assert_in_bounds(self, points) -> None:
        for point in points:
            self.assertGreaterEqual(point.x, 0.0)
            self.assertGreaterEqual(point.y, 0.0)
            self.assertLessEqual(point.x, 500.0)
            self.assertLessEqual(point.y, 500.0)

    def _assert_star_alternating_radii(
        self,
        points,
        center_x: float,
        center_y: float,
        outer: float,
        inner: float,
    ) -> None:
        for i, point in enumerate(points):
            radius = hypot(point.x - center_x, point.y - center_y)
            target = outer if i % 2 == 0 else inner
            self.assertLess(abs(radius - target), 3.0)

    def _is_concave(self, points) -> bool:
        signs: set[int] = set()
        n = len(points)
        for i in range(n):
            a = points[i]
            b = points[(i + 1) % n]
            c = points[(i + 2) % n]
            cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if cross > 0:
                signs.add(1)
            elif cross < 0:
                signs.add(-1)
        return len(signs) > 1


if __name__ == "__main__":
    unittest.main()
