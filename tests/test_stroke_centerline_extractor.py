import unittest

import _bootstrap  # noqa: F401
from src.infrastructure.processing.stroke_centerline_extractor import extract_stroke_centerlines


class TestStrokeCenterlineExtractor(unittest.TestCase):
    def test_extracts_open_polyline_from_thick_open_stroke(self) -> None:
        mask = [[0 for _ in range(9)] for _ in range(9)]
        for y in range(2, 7):
            for x in range(3, 6):
                mask[y][x] = 1

        contours = extract_stroke_centerlines(mask)

        self.assertEqual(len(contours), 1)
        contour = contours[0]
        self.assertNotEqual(contour[0], contour[-1])
        self.assertGreaterEqual(len(contour), 2)

    def test_extracts_closed_polyline_from_loop(self) -> None:
        mask = [[0 for _ in range(10)] for _ in range(10)]
        for x in range(2, 8):
            mask[2][x] = 1
            mask[7][x] = 1
        for y in range(2, 8):
            mask[y][2] = 1
            mask[y][7] = 1

        contours = extract_stroke_centerlines(mask)

        self.assertEqual(len(contours), 1)
        contour = contours[0]
        self.assertEqual(contour[0], contour[-1])


if __name__ == "__main__":
    unittest.main()
