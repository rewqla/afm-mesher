r"""Зображення через обидва шляхи: UI (завантаження PNG → контури canvas → адаптер) і adapter.run(QImage).

    .venv\Scripts\python diagnostics\image_paths_probe.py [--mode balanced] [png ...]

Для кожного: контури (замкнені / відкриті), кількість дірок (замкнених контурів усередині області),
результат тріангуляції (трикутники, ребра розрізів, попередження).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.tri_debug as tri_debug_module  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from diagnostics.ui_image_probe import ui_contours  # noqa: E402
from src.presentation.canvas import Canvas  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402

OUTPUT_DIR = ROOT / "diagnostics" / "output"
DEFAULT_IMAGES = [*sorted((ROOT / "data/images").glob("*.png")), *(ROOT / f"diagnostics/cases/ui_bugs/{n}.png" for n in ("1", "1_clean"))]


def describe(contours) -> str:
    closed = sum(1 for c in contours if len(c) >= 3 and tuple(c[0]) == tuple(c[-1]))
    return f"closed={closed} open={len(contours) - closed}"


def summary(mesh) -> str:
    cut_edges = len(mesh.cut_edges)
    idx = mesh.indexed_mesh()
    boundary_loops = _boundary_loops(idx)
    warnings = len(getattr(mesh, "warnings", ()) or ())
    return f"OK tris={len(mesh.triangles)} loops={boundary_loops} cut_edges={cut_edges} warnings={warnings}"


def _boundary_loops(idx) -> int:
    """Boundary edges (one triangle) grouped into connected chains, cut sides excluded by twin numbering."""
    from collections import Counter, defaultdict

    edges = Counter()
    for t in idx.triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[frozenset((a, b))] += 1
    coords = {i + idx.index_base: (round(x, 6), round(y, 6)) for i, (x, y) in enumerate(idx.nodes)}
    # Glue twins by coordinate; boundary edges on both sides of a cut cancel out.
    glued = Counter()
    for e, n in edges.items():
        if n == 1:
            a, b = (coords[v] for v in e)
            glued[frozenset((a, b))] += 1
    graph = defaultdict(set)
    for e, n in glued.items():
        if n == 1 and len(e) == 2:
            a, b = tuple(e)
            graph[a].add(b)
            graph[b].add(a)
    seen = set()
    loops = 0
    for start in graph:
        if start in seen:
            continue
        loops += 1
        stack = [start]
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(graph[p] - seen)
    return loops


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("pngs", nargs="*")
    parser.add_argument("--mode", default="balanced")
    args = parser.parse_args()
    log = OUTPUT_DIR / "afm_debug_image_paths.log"
    afm_module._AFM_DEBUG_LOG_PATH = log
    tri_debug_module._LOG_PATH = log
    mode = TriangulationMode(args.mode)
    for png in [Path(p) for p in args.pngs] or DEFAULT_IMAGES:
        window, _image, contours = ui_contours(png)
        adapter = window._triangulation_adapter  # noqa: SLF001
        try:
            ui = summary(adapter.run_from_contours(contours, mode=mode, inclusion_mode=window.inclusion_mode())[0])
        except Exception as error:  # noqa: BLE001
            ui = f"FAIL {error}"
        binary = Canvas.binarize_image(QImage(str(png)), threshold=127)
        try:
            direct = summary(adapter.run(binary, mode=mode)[0])
        except Exception as error:  # noqa: BLE001
            direct = f"FAIL {error}"
        print(f"{png.name:22} UI[{describe(contours)}] {ui}\n{'':22} run(QImage) {direct}", flush=True)


if __name__ == "__main__":
    main()
