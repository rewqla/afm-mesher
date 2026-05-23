import unittest

import _bootstrap  # noqa: F401
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.presentation.canvas import Canvas
from src.presentation.tools import Tool


class TestCanvasContourIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_second_append_keeps_open_endpoint_on_opposite_side(self) -> None:
        canvas = Canvas()
        canvas.set_geometry_contours([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])

        base, base_index, attach_side, anchor = canvas._take_continuation_target(0.2, 0.1)  # noqa: SLF001
        self.assertEqual(attach_side, "start")
        self.assertEqual(anchor, (0.0, 0.0))
        self.assertEqual(base_index, 0)

        canvas._active_base_contour = base  # noqa: SLF001
        canvas._active_base_contour_index = base_index  # noqa: SLF001
        canvas._active_attach_side = attach_side  # noqa: SLF001
        canvas._current_stroke = [anchor, (-2.0, 3.0), (-3.0, 8.0)]  # noqa: SLF001
        canvas._maybe_commit_pen_contour()  # noqa: SLF001

        contour = canvas.geometry_contours()[0]
        self.assertEqual(contour[-1], (10.0, 10.0))
        self.assertNotEqual(contour[0], (0.0, 0.0))

    def test_short_pen_continuation_restores_original_open_contour(self) -> None:
        canvas = Canvas()
        original = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        canvas.set_geometry_contours([original], preprocess=False, redraw_image=False, emit_change=False)

        base, base_index, attach_side, anchor = canvas._take_continuation_target(10.0, 10.0)  # noqa: SLF001
        self.assertEqual(canvas.geometry_contours(), [original])

        canvas._active_base_contour = base  # noqa: SLF001
        canvas._active_base_contour_index = base_index  # noqa: SLF001
        canvas._active_attach_side = attach_side  # noqa: SLF001
        canvas._current_stroke = [anchor]  # noqa: SLF001
        canvas._maybe_commit_pen_contour()  # noqa: SLF001

        self.assertEqual(canvas.geometry_contours(), [original])

    def test_take_continuation_target_does_not_remove_contour_from_cache(self) -> None:
        canvas = Canvas()
        original = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        canvas.set_geometry_contours([original], preprocess=False, redraw_image=False, emit_change=False)

        base, base_index, attach_side, anchor = canvas._take_continuation_target(10.0, 10.0)  # noqa: SLF001

        self.assertEqual(base, original)
        self.assertEqual(base_index, 0)
        self.assertEqual(attach_side, "end")
        self.assertEqual(anchor, (10.0, 10.0))
        self.assertEqual(canvas.geometry_contours(), [original])

    def test_invalid_segment_visualization_keeps_only_latest_hint(self) -> None:
        canvas = Canvas()
        canvas.set_invalid_segments([((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0))])

        self.assertEqual(canvas._invalid_segments, [((2.0, 2.0), (3.0, 3.0))])  # noqa: SLF001

    def test_invalid_point_visualization_replaces_previous_points(self) -> None:
        canvas = Canvas()

        canvas.set_invalid_points([(10.0, 10.0), (20.0, 30.0)])

        self.assertEqual(canvas._invalid_points, [(10.0, 10.0), (20.0, 30.0)])  # noqa: SLF001

    def test_eraser_marks_geometry_as_dirty_and_clears_cached_contours(self) -> None:
        canvas = Canvas(width=40, height=40)
        canvas.set_geometry_contours(
            [[(5.0, 5.0), (20.0, 5.0), (20.0, 20.0), (5.0, 5.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        canvas.set_tool(Tool.ERASER)

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10.0, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mousePressEvent(press)

        self.assertTrue(canvas.geometry_is_dirty())
        self.assertEqual(canvas.geometry_contours(), [])

    def test_segment_tool_preserves_geometry_cache_and_adds_open_segment(self) -> None:
        canvas = Canvas(width=80, height=80)
        canvas.set_geometry_contours(
            [[(5.0, 5.0), (30.0, 5.0), (30.0, 30.0), (5.0, 5.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        canvas._tool = Tool.SEGMENT  # noqa: SLF001
        canvas._shape_start = QPoint(40, 40)  # noqa: SLF001
        canvas._shape_end = QPoint(60, 60)  # noqa: SLF001

        canvas._commit_shape()  # noqa: SLF001

        contours = canvas.geometry_contours()
        self.assertFalse(canvas.geometry_is_dirty())
        self.assertEqual(len(contours), 2)
        self.assertEqual(contours[-1], [(40.0, 40.0), (60.0, 60.0)])

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

    def test_resize_canvas_allows_growing_blank_canvas(self) -> None:
        canvas = Canvas(width=300, height=300)

        success, message = canvas.resize_canvas(640, 480, record_history=False)

        self.assertTrue(success)
        self.assertIsNone(message)
        self.assertEqual(canvas.canvas_size().width(), 640)
        self.assertEqual(canvas.canvas_size().height(), 480)

    def test_resize_canvas_blocks_when_drawn_content_would_overflow(self) -> None:
        canvas = Canvas(width=300, height=300)
        canvas.set_geometry_contours(
            [[(20.0, 20.0), (280.0, 20.0), (280.0, 40.0), (20.0, 40.0), (20.0, 20.0)]],
            preprocess=False,
            redraw_image=True,
            emit_change=False,
        )

        success, message = canvas.resize_canvas(256, 256, record_history=False)

        self.assertFalse(success)
        self.assertIsNotNone(message)
        self.assertIn("would fall outside the canvas", message)
        self.assertEqual(canvas.canvas_size().width(), 300)
        self.assertEqual(canvas.canvas_size().height(), 300)

    def test_resize_canvas_blocks_when_hidden_geometry_would_overflow(self) -> None:
        canvas = Canvas(width=300, height=300)
        canvas.set_geometry_contours(
            [[(10.0, 10.0), (290.0, 10.0), (290.0, 290.0), (10.0, 290.0), (10.0, 10.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )

        success, message = canvas.resize_canvas(280, 280, record_history=False)

        self.assertFalse(success)
        self.assertIsNotNone(message)
        self.assertIn("x=10..290", message)


if __name__ == "__main__":
    unittest.main()
