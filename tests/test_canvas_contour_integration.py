import unittest

import _bootstrap  # noqa: F401
from PySide6.QtWidgets import QApplication

from src.presentation.canvas import Canvas


class TestCanvasContourIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_second_append_keeps_open_endpoint_on_opposite_side(self) -> None:
        canvas = Canvas()
        canvas.set_geometry_contours([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])

        base, attach_side, anchor = canvas._take_continuation_target(0.2, 0.1)  # noqa: SLF001
        self.assertEqual(attach_side, "start")
        self.assertEqual(anchor, (0.0, 0.0))

        canvas._active_base_contour = base  # noqa: SLF001
        canvas._active_attach_side = attach_side  # noqa: SLF001
        canvas._current_stroke = [anchor, (-2.0, 3.0), (-3.0, 8.0)]  # noqa: SLF001
        canvas._maybe_commit_pen_contour()  # noqa: SLF001

        contour = canvas.geometry_contours()[0]
        self.assertEqual(contour[-1], (10.0, 10.0))
        self.assertNotEqual(contour[0], (0.0, 0.0))

    def test_invalid_segment_visualization_keeps_only_latest_hint(self) -> None:
        canvas = Canvas()
        canvas.set_invalid_segments([((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0))])

        self.assertEqual(canvas._invalid_segments, [((2.0, 2.0), (3.0, 3.0))])  # noqa: SLF001

    def test_syncing_geometry_contours_without_redraw_preserves_loaded_image(self) -> None:
        canvas = Canvas(width=20, height=20)
        original = canvas.image_data()
        original_pixel = original.pixelColor(10, 10)

        canvas.set_geometry_contours(
            [[(2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0), (2.0, 2.0)]],
            redraw_image=False,
            emit_change=False,
        )

        current = canvas.image_data()
        self.assertEqual(current.size(), original.size())
        self.assertEqual(current.pixelColor(10, 10), original_pixel)
        self.assertEqual(len(canvas.geometry_contours()), 1)

    def test_syncing_loaded_image_contours_without_preprocess_keeps_closed_boundary(self) -> None:
        canvas = Canvas(width=20, height=20)
        canvas.set_geometry_contours(
            [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )

        contour = canvas.geometry_contours()[0]
        self.assertEqual(contour[0], contour[-1])


if __name__ == "__main__":
    unittest.main()
