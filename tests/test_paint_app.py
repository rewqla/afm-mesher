import unittest

import _bootstrap  # noqa: F401
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from src.presentation.paint_app import PaintApp, TRIANGULATION_INFO_DIALOG_TEXT
from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.presentation.tools import Tool


class _BranchDetectingAdapter:
    def __init__(self) -> None:
        self.branch_detection_calls = 0

    def detect_stroke_branch_points_from_image(self, _image):  # noqa: ANN001
        self.branch_detection_calls += 1
        return [(5, 5)]


class _ContourExtractionProbeAdapter:
    def __init__(self) -> None:
        self.extract_contours_calls = 0
        self.extract_stroke_calls = 0

    def extract_contours_from_image(self, _image):  # noqa: ANN001
        self.extract_contours_calls += 1
        raise ValueError("no contours")

    def extract_stroke_contours_from_image(self, _image):  # noqa: ANN001
        self.extract_stroke_calls += 1
        raise ValueError("no strokes")


class TestPaintApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_window_resizes_with_canvas_delta(self) -> None:
        window = PaintApp()
        initial_window_size = window.size()

        window._resize_window_for_canvas_change(900, 650, 640, 480)  # noqa: SLF001

        self.assertEqual(window.width(), initial_window_size.width() - 260)
        self.assertEqual(window.height(), initial_window_size.height() - 170)

    def test_resolve_custom_settings_includes_meters_per_pixel(self) -> None:
        window = PaintApp()
        for index in range(window._mode_combo.count()):  # noqa: SLF001
            if window._mode_combo.itemData(index) == "custom":  # noqa: SLF001
                window._mode_combo.setCurrentIndex(index)  # noqa: SLF001
                break
        window._meters_per_pixel_spin.setValue(0.01)  # noqa: SLF001

        mode, settings = window._resolve_mode_and_settings()  # noqa: SLF001

        self.assertEqual(mode.value, "custom")
        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertAlmostEqual(settings.meters_per_pixel, 0.01, places=12)

    def test_build_triangulation_info_dialog_uses_expected_text_and_button(self) -> None:
        window = PaintApp()

        dialog = window._build_triangulation_info_dialog()  # noqa: SLF001

        self.assertEqual(dialog.windowTitle(), "Інформація про тріангуляцію")
        self.assertEqual(dialog.text(), TRIANGULATION_INFO_DIALOG_TEXT)
        ok_button = dialog.button(dialog.StandardButton.Ok)
        self.assertIsNotNone(ok_button)
        assert ok_button is not None
        self.assertEqual(ok_button.text(), "Продовжити")

    def test_display_triangulation_result_updates_quality_and_node_difference_labels(self) -> None:
        window = PaintApp()
        mesh = Mesh(
            triangles=[
                Triangle(
                    a=Point(0.0, 0.0),
                    b=Point(1.0, 0.0),
                    c=Point(0.0, 1.0),
                )
            ],
            indexed_mesh_data=IndexedMesh(
                nodes=[
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (0.25, 0.25),
                    (0.5, 0.5),
                    (0.75, 0.75),
                    (1.0, 1.0),
                ],
                triangles=[(1, 7, 4)],
                boundary_nodes=set(),
                interface_nodes=set(),
                bandwidth=6,
                index_base=1,
            ),
        )

        window.display_triangulation_result(mesh, 0.125)

        self.assertIn("0.1250", window.statusBar().currentMessage())
        self.assertIn("вузлів=7", window.statusBar().currentMessage())
        self.assertIn("6", window.statusBar().currentMessage())

    def test_branch_detection_is_skipped_for_multiple_contours(self) -> None:
        window = PaintApp()
        adapter = _BranchDetectingAdapter()
        window._triangulation_adapter = adapter  # noqa: SLF001
        window._contours_for_triangulation = lambda *_args, **_kwargs: [  # noqa: SLF001
            [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
            [(10, 10), (12, 12)],
        ]
        window._boundary_validator.select_outer_contour = lambda _contours: None  # noqa: SLF001
        open_issue = type(
            "Issue",
            (),
            {"segment": ((10, 10), (12, 12)), "contour_index": 1},
        )()
        window._boundary_validator.validate = lambda _contours: type(  # noqa: SLF001
            "Validation",
            (),
            {"is_valid": False, "open_contours": [open_issue], "self_intersections": []},
        )()

        original_information = QMessageBox.information
        QMessageBox.information = lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok
        try:
            window._on_triangulation()  # noqa: SLF001
        finally:
            QMessageBox.information = original_information

        self.assertEqual(adapter.branch_detection_calls, 0)

    def test_undo_prefers_boundary_contour_extraction_over_stroke_centerlines(self) -> None:
        window = PaintApp()
        adapter = _ContourExtractionProbeAdapter()
        window._triangulation_adapter = adapter  # noqa: SLF001

        image = QImage(40, 40, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.black)
        window._canvas.set_image(image, record_history=True)  # noqa: SLF001
        self.assertTrue(window._canvas.undo())  # noqa: SLF001

        window._contours_for_triangulation(window._canvas.image_data(), prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(adapter.extract_contours_calls, 1)
        self.assertEqual(adapter.extract_stroke_calls, 0)

    def test_redo_prefers_boundary_contour_extraction_over_stroke_centerlines(self) -> None:
        window = PaintApp()
        adapter = _ContourExtractionProbeAdapter()
        window._triangulation_adapter = adapter  # noqa: SLF001

        image = QImage(40, 40, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.black)
        window._canvas.set_image(image, record_history=True)  # noqa: SLF001
        self.assertTrue(window._canvas.undo())  # noqa: SLF001
        self.assertTrue(window._canvas.redo())  # noqa: SLF001

        window._contours_for_triangulation(window._canvas.image_data(), prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(adapter.extract_contours_calls, 1)
        self.assertEqual(adapter.extract_stroke_calls, 0)

    def test_erase_prefers_boundary_contour_extraction_over_stroke_centerlines(self) -> None:
        window = PaintApp()
        adapter = _ContourExtractionProbeAdapter()
        window._triangulation_adapter = adapter  # noqa: SLF001
        window._canvas.set_tool(Tool.ERASER)  # noqa: SLF001

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10.0, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window._canvas.mousePressEvent(press)  # noqa: SLF001

        window._contours_for_triangulation(window._canvas.image_data(), prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(adapter.extract_contours_calls, 1)
        self.assertEqual(adapter.extract_stroke_calls, 0)


if __name__ == "__main__":
    unittest.main()
