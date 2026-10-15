"""Проганяє PNG (або JSON намальованих контурів) через той самий шлях, що UI (завантаження зображення → контури canvas → адаптер).

    .venv\\Scripts\\python diagnostics\\ui_image_probe.py diagnostics\\cases\\ui_bugs\\1.png --mode balanced
    .venv\\Scripts\\python diagnostics\\ui_image_probe.py <png> --dump contours.json   # лише зберегти контури

Друкує контури (замкнені/відкриті), результат тріангуляції, таймінги етапів і підсумок
ключових подій з afm-логу (diagnostics/output/afm_debug_<ім'я>.log).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.tri_debug as tri_debug_module  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.presentation.canvas import Canvas  # noqa: E402
from src.presentation.paint_app import PaintApp  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def ui_contours(png: Path) -> tuple[PaintApp, QImage, list]:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    window = PaintApp()
    if png.suffix.lower() == ".json":
        # Намальована геометрія: {"contours": [[[x, y], ...], ...]} як у canvas.geometry_contours().
        drawn = json.loads(png.read_text(encoding="utf-8"))["contours"]
        window._canvas.set_geometry_contours(  # noqa: SLF001
            [[(float(x), float(y)) for x, y in c] for c in drawn], preprocess=False, redraw_image=True, emit_change=False
        )
    else:
        image = QImage(str(png))
        binary = Canvas.binarize_image(image, threshold=127)
        window._canvas.set_image(binary)  # noqa: SLF001
        window._refresh_geometry_from_canvas_image(binary, prefer_strokes=True)  # noqa: SLF001
    image_data = window._canvas.image_data()  # noqa: SLF001
    contours = window._contours_for_triangulation(image_data, prefer_strokes=True)  # noqa: SLF001
    messages: list[str] = []
    QMessageBox.information = lambda *a, **k: messages.append(str(a[-1])) or QMessageBox.StandardButton.Ok
    QMessageBox.warning = lambda *a, **k: messages.append(str(a[-1])) or QMessageBox.StandardButton.Ok
    prepared = window._prepare_contours_for_triangulation(contours) if contours else contours  # noqa: SLF001
    for m in messages:
        print("  UI message:", m)
    return window, image_data, prepared or []


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("png")
    parser.add_argument("--mode", default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--budget", type=float, default=None, help="бюджет часу адаптера, с")
    parser.add_argument("--dump", default=None, help="зберегти контури в JSON і вийти")
    args = parser.parse_args()

    png = Path(args.png)
    log = OUTPUT_DIR / f"afm_debug_{png.stem}.log"
    OUTPUT_DIR.mkdir(exist_ok=True)
    log.write_text("", encoding="utf-8")
    afm_module._AFM_DEBUG_LOG_PATH = log
    tri_debug_module._LOG_PATH = log

    window, image_data, contours = ui_contours(png)
    print(f"contours: {len(contours)}")
    for i, c in enumerate(contours):
        closed = len(c) >= 3 and tuple(c[0]) == tuple(c[-1])
        print(f"  #{i}: {'closed' if closed else 'open  '} points={len(c)} first={tuple(c[0])} last={tuple(c[-1])}")
    if args.dump:
        Path(args.dump).write_text(json.dumps([[list(p) for p in c] for c in contours]), encoding="utf-8")
        return

    adapter = window._triangulation_adapter  # noqa: SLF001
    if args.budget is not None:
        adapter.time_budget_s = args.budget
    try:
        mesh, coefficient = adapter.run_from_contours(
            contours, mode=TriangulationMode(args.mode), inclusion_mode=window.inclusion_mode()
        )
        t = mesh.timings
        print(f"OK triangles={len(mesh.triangles)} quality={coefficient:.3f} cut_edges={len(mesh.cut_edges)}")
        print(f"timings: total={t.total_s:.2f} contours={t.contours_s:.2f} afm={t.afm_s:.2f} "
              f"post={t.postprocess_s:.2f} numbering={t.numbering_s:.2f}")
        for w in getattr(mesh, "warnings", ()) or ():
            print("  WARNING:", w)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {type(exc).__name__}: {exc}")

    events = Counter()
    for line in log.read_text(encoding="utf-8").splitlines():
        if "[" in line:
            events[line.split("[", 1)[1].split("]", 1)[0]] += 1
    print("log events:", dict(events))
    for line in log.read_text(encoding="utf-8").splitlines():
        if any(k in line for k in ("single_pass.start", "mesh_fail", "mesh_success", "geometry]", "overflow",
                                   "time_budget", "stall_diag", "final_validation", "cut_drop", "warning")):
            print("  ", line[24:400])


if __name__ == "__main__":
    main()
