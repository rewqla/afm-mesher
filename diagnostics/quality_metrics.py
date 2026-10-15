r"""Метрики дрібних / гострих трикутників по кейсах (задача «дрібні трикутники вздовж кривих меж»).

Запуск з кореня проєкту:
    .venv\Scripts\python diagnostics\quality_metrics.py [--mode balanced ...] [--only name ...]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diagnostics.perf_reference as perf_reference  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402

CASES = {
    "bug3_drawn": ROOT / "diagnostics/cases/ui_bugs/bug3_drawn_contours.json",
    "bug1_drawn": ROOT / "diagnostics/cases/ui_bugs/bug1_drawn_contours.json",
    "fig22_handdrawn": ROOT / "diagnostics/cases/fig22_handdrawn.json",
    "img_circle": ROOT / "data/images/circle.png",
    "img_spot_with_circle": ROOT / "data/images/spot_with_circle.png",
    "img_ui_bug1": ROOT / "diagnostics/cases/ui_bugs/1.png",
    "img_ui_bug3": ROOT / "diagnostics/cases/ui_bugs/3.png",
}


def triangle_angles(a, b, c) -> list[float]:
    def ang(p, q, r):
        v1 = (q.x - p.x, q.y - p.y)
        v2 = (r.x - p.x, r.y - p.y)
        n = math.hypot(*v1) * math.hypot(*v2)
        if n == 0:
            return 0.0
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
        return math.degrees(math.acos(cos))

    return [ang(a, b, c), ang(b, c, a), ang(c, a, b)]


def metrics(mesh) -> dict:
    areas = []
    min_angles = []
    qualities = []
    for t in mesh.triangles:
        a, b, c = t.a, t.b, t.c
        area = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2.0
        areas.append(area)
        min_angles.append(min(triangle_angles(a, b, c)))
        s = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (b.x - c.x) ** 2 + (b.y - c.y) ** 2 + (c.x - a.x) ** 2 + (c.y - a.y) ** 2
        qualities.append(4.0 * math.sqrt(3.0) * area / s if s else 0.0)
    median = statistics.median(areas)
    return {
        "tris": len(areas),
        "min_area": min(areas),
        "median_area": median,
        "min_over_median": min(areas) / median,
        "angles_lt10": sum(a < 10.0 for a in min_angles),
        "angles_lt15": sum(a < 15.0 for a in min_angles),
        "min_angle": min(min_angles),
        "q_lt01": sum(q < 0.1 for q in qualities),
        "cuts": len({frozenset(((a.x, a.y), (b.x, b.y))) for a, b in mesh.cut_edges}),
        "time": mesh.timings.total_s if mesh.timings else 0.0,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--mode", nargs="*", default=["fast", "balanced", "accurate"])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication(["metrics"])  # noqa: F841
    perf_reference.CASES.update(CASES)
    results = {}
    print(f"{'case':22} {'mode':9} {'tris':>5} {'minA':>7} {'medA':>7} {'min/med':>8} "
          f"{'<10°':>5} {'<15°':>5} {'minAng':>7} {'q<.1':>5} {'time':>6}")
    for name in args.only or CASES:
        for mode in args.mode:
            try:
                mesh, _ = perf_reference.run(name, TriangulationMode(mode))
            except Exception as error:  # noqa: BLE001
                print(f"{name:22} {mode:9} FAILED: {error}")
                continue
            m = metrics(mesh)
            results[f"{name}/{mode}"] = m
            print(f"{name:22} {mode:9} {m['tris']:5} {m['min_area']:7.2f} {m['median_area']:7.1f} "
                  f"{m['min_over_median']:8.4f} {m['angles_lt10']:5} {m['angles_lt15']:5} {m['min_angle']:7.2f} "
                  f"{m['q_lt01']:5} {m['time']:6.2f}", flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
