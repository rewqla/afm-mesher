import unittest

import _bootstrap  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from src.application.services.boundary_validator import BoundaryValidator
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

    def _blank_image(self) -> QImage:
        image = QImage(120, 120, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        return image


if __name__ == "__main__":
    unittest.main()
