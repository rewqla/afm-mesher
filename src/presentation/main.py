from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.presentation.paint_app import PaintApp


def launch_ui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = PaintApp()
    window.show()
    return app.exec()


def main() -> None:
    raise SystemExit(launch_ui())


if __name__ == "__main__":
    main()
