import unittest

import _bootstrap  # noqa: F401
from src.domain.entities.point import Point
from src.domain.geometry.contour_pipeline import (
    PreprocessContour,
    SmartAppendContour,
    ValidateContour,
    ValidationResult,
    preprocess_contour,
    smart_append_contour,
    validate_contour,
)


class TestContourPipeline(unittest.TestCase):
    def test_preprocess_clips_tail_and_marks_closed(self) -> None:
        raw = [
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10),
            Point(0.3, 0.3),
            Point(2, -2),
            Point(3, -3),
        ]

        processed = preprocess_contour(raw, epsilon=0.5, closure_tolerance=1.0)
        result = validate_contour(processed)

        self.assertEqual(processed[-1], processed[0])
        self.assertEqual(len(processed), 5)
        self.assertTrue(result.is_closed)
        self.assertTrue(result.is_valid)

    def test_smart_append_smooths_reverse_snap_corner(self) -> None:
        contour = [Point(0, 0), Point(10, 0), Point(20, 0)]
        new_stroke = [Point(20.2, 0.1), Point(19.0, 2.0), Point(22.0, 4.0), Point(25.0, 4.0)]

        appended = smart_append_contour(contour, new_stroke, snap_tolerance=1.0)

        self.assertEqual(appended[2], Point(20, 0))
        self.assertGreaterEqual(appended[3].x, 20.0)
        self.assertGreaterEqual(appended[4].x, appended[3].x)

    def test_validate_detects_true_self_intersection(self) -> None:
        contour = [
            Point(0, 0),
            Point(10, 10),
            Point(0, 10),
            Point(10, 0),
            Point(0, 0),
        ]

        result = validate_contour(contour)

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_closed)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.intersect_point)

    def test_smart_append_accepts_short_corrective_stroke(self) -> None:
        contour = [Point(0, 0), Point(10, 0), Point(10, 6)]
        short_stroke = [Point(10.2, 6.1), Point(10.1, 9.5)]

        appended = smart_append_contour(contour, short_stroke, snap_tolerance=1.0)
        processed = preprocess_contour(appended, epsilon=0.5, closure_tolerance=1.0)

        self.assertGreater(len(appended), len(contour))
        self.assertGreater(processed[-1].y, contour[-1].y)

    def test_smart_append_at_start_preserves_original_orientation(self) -> None:
        contour = [Point(0, 0), Point(10, 0), Point(10, 10)]
        new_stroke = [Point(0.2, 0.1), Point(-2, 4), Point(-3, 8)]

        appended = smart_append_contour(contour, new_stroke, snap_tolerance=1.0, attach_to="start")

        self.assertEqual(appended[-1], contour[-1])
        self.assertNotEqual(appended[0], contour[0])
        self.assertEqual(appended[-2], contour[-2])

    def test_preprocess_removes_duplicate_points_and_micro_jitter(self) -> None:
        raw = [
            Point(0, 0),
            Point(0.1, 0.1),
            Point(5, 0.2),
            Point(10, 0),
            Point(10.1, 0.1),
            Point(10, 10),
            Point(0, 10),
            Point(0, 0),
        ]

        processed = preprocess_contour(raw, epsilon=0.5, closure_tolerance=1.0)

        self.assertLess(len(processed), len(raw))
        self.assertEqual(processed[0], processed[-1])

    def test_public_api_aliases_match_core_functions(self) -> None:
        raw = [Point(0, 0), Point(5, 0), Point(5, 5), Point(0, 5), Point(0.2, 0.2)]
        processed = PreprocessContour(raw, epsilon=0.5, closureTolerance=1.0)
        validation = ValidateContour(processed)
        appended = SmartAppendContour(
            [Point(0, 0), Point(4, 0)],
            [Point(4.1, 0.0), Point(6, 0)],
            snapTolerance=1.0,
        )

        self.assertEqual(processed, preprocess_contour(raw, epsilon=0.5, closure_tolerance=1.0))
        self.assertEqual(validation, validate_contour(processed))
        self.assertEqual(
            appended,
            smart_append_contour([Point(0, 0), Point(4, 0)], [Point(4.1, 0.0), Point(6, 0)], snap_tolerance=1.0),
        )

    def test_preprocess_preserves_explicitly_closed_image_contour(self) -> None:
        raw = [
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10),
            Point(0, 0),
        ]

        processed = preprocess_contour(raw, epsilon=0.5, closure_tolerance=8.0)

        self.assertEqual(processed[0], processed[-1])
        self.assertGreaterEqual(len(processed), 4)


if __name__ == "__main__":
    unittest.main()
