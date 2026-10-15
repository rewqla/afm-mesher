r"""Час відмальовки сітки на canvas (найбільша сітка набору) і хеш пікселів результату.

Запуск з кореня проєкту:
    .venv\Scripts\python diagnostics\perf_canvas.py [case] [mode]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import diagnostics.perf_reference as perf_reference  # noqa: E402
from src.presentation.canvas import Canvas  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    case = sys.argv[1] if len(sys.argv) > 1 else "img_spot_with_circle"
    mode = TriangulationMode(sys.argv[2] if len(sys.argv) > 2 else "accurate")
    app = QApplication.instance() or QApplication(["perf"])  # noqa: F841
    mesh, _ = perf_reference.run(case, mode)
    canvas = Canvas(900, 900)
    canvas.set_mesh_overlay(mesh)
    target = QImage(900, 900, QImage.Format.Format_ARGB32)
    rounds = 50
    best = float("inf")
    for _ in range(5):
        t0 = perf_counter()
        for _ in range(rounds):
            painter = QPainter(target)
            canvas._draw_mesh_overlay(painter, mesh)  # noqa: SLF001
            painter.end()
        best = min(best, (perf_counter() - t0) / rounds)
    image = canvas.rendered_image()
    digest = hashlib.sha256(bytes(image.constBits())).hexdigest()[:16]
    print(f"{case}/{mode.value}: трикутників={len(mesh.triangles)}, "
          f"відмальовка сітки={best * 1000:.2f} мс/кадр, хеш пікселів={digest}")


if __name__ == "__main__":
    main()
