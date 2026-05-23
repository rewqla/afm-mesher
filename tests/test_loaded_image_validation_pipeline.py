import unittest

import _bootstrap  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from src.application.services.boundary_validator import BoundaryValidator
from src.presentation.tools import Tool
from src.presentation.paint_app import PaintApp
from src.presentation.triangulation_adapter import TriangulationAdapter


class TestLoadedImageValidationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.adapter = TriangulationAdapter()
        self.validator = BoundaryValidator()

    def test_open_loaded_stroke_is_reported_as_open_contour(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        contours = self.adapter.extract_stroke_contours_from_image(image)
        result = self.validator.validate(contours)

        self.assertGreaterEqual(len(contours), 1)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def test_closed_loaded_loop_is_reported_as_valid(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawEllipse(20, 20, 80, 80)
        painter.end()

        contours = self.adapter.extract_stroke_contours_from_image(image)
        result = self.validator.validate(contours)

        self.assertGreaterEqual(len(contours), 1)
        self.assertTrue(result.is_valid)

    def test_branching_loaded_stroke_reports_branch_points(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawLine(15, 60, 105, 60)
        painter.drawLine(60, 15, 60, 105)
        painter.end()

        branch_points = self.adapter.detect_stroke_branch_points_from_image(image)

        self.assertGreaterEqual(len(branch_points), 1)

    def test_paint_app_syncs_loaded_image_contours_into_canvas_geometry(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        contours = app._refresh_geometry_from_canvas_image(image, prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(len(contours), 1)
        self.assertEqual(contours, app._canvas.geometry_contours())  # noqa: SLF001

    def test_paint_app_refreshes_dirty_geometry_cache_before_triangulation(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        app._canvas.set_geometry_contours(  # noqa: SLF001
            [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        app._canvas._geometry_dirty = True  # noqa: SLF001

        contours = app._contours_for_triangulation(image, prefer_strokes=True)  # noqa: SLF001
        result = self.validator.validate(contours)

        self.assertGreaterEqual(len(contours), 1)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def test_switching_back_to_pen_resyncs_dirty_geometry_from_canvas_image(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        app._canvas.set_image(image, record_history=False)  # noqa: SLF001
        app._canvas._geometry_contours = []  # noqa: SLF001
        app._canvas._geometry_dirty = True  # noqa: SLF001

        app._set_tool(Tool.PEN)  # noqa: SLF001

        contours = app._canvas.geometry_contours()  # noqa: SLF001
        result = self.validator.validate(contours)
        self.assertGreaterEqual(len(contours), 1)
        self.assertFalse(app._canvas.geometry_is_dirty())  # noqa: SLF001
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def _blank_image(self) -> QImage:
        image = QImage(120, 120, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        return image


if __name__ == "__main__":
    unittest.main()
