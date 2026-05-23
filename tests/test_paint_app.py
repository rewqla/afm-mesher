import unittest

import _bootstrap  # noqa: F401
from PySide6.QtWidgets import QApplication

from src.presentation.paint_app import PaintApp


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
        window._mode_combo.setCurrentText("Custom")  # noqa: SLF001
        window._meters_per_pixel_spin.setValue(0.01)  # noqa: SLF001

        mode, settings = window._resolve_mode_and_settings()  # noqa: SLF001

        self.assertEqual(mode.value, "custom")
        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertAlmostEqual(settings.meters_per_pixel, 0.01, places=12)


if __name__ == "__main__":
    unittest.main()
