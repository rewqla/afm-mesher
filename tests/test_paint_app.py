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


if __name__ == "__main__":
    unittest.main()
