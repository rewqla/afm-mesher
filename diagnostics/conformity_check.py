r"""Конформність сіток perf_reference-кейсів (накладання, висячі вузли, береги розрізів) + якість.

    .venv\Scripts\python diagnostics\conformity_check.py [--only name ...] [--mode fast ...]

Вхід мешера (межа, дірки, розрізи) перехоплюється, і сітка перевіряється analyze_conformity з
tests/test_mesh_conformity_invariants.py. «cut_edge_unpaired» біля перетину двох розрізів
перевіряється окремо відносно прямої самого ребра (див. tests/test_curves_and_image_strokes.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnostics.perf_reference as perf_reference  # noqa: E402
import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.triangulation_adapter as adapter_module  # noqa: E402
import test_mesh_conformity_invariants as inv  # noqa: E402
from diagnostics.quality_metrics import metrics  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402


def check(name: str, mode: TriangulationMode) -> tuple[list[str], dict]:
    captured: dict[str, tuple] = {}

    class RecordingMesher(afm_module.AdvancingFrontMesher):
        def generate_with_holes_and_cuts(self, boundary, holes, cuts, **kwargs):
            mesh = super().generate_with_holes_and_cuts(boundary, holes, cuts, **kwargs)
            captured["input"] = (list(boundary), [list(h) for h in holes], [list(c) for c in cuts])
            return mesh

    with patch.object(adapter_module, "AdvancingFrontMesher", RecordingMesher):
        mesh, _ = perf_reference.run(name, mode)
    boundary, holes, cuts = captured["input"]
    tris = [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in mesh.triangles]
    shell = [(p.x, p.y) for p in boundary]
    cut_lines = [[(p.x, p.y) for p in c] for c in cuts]
    report = inv.analyze_conformity(
        tris, shell, [[(p.x, p.y) for p in h] for h in holes], inv.CutGeometry.build(cut_lines, shell)
    )
    problems = inv.overlap_violations(report)
    if report["hanging"]:
        problems.append(f"hanging: {len(report['hanging'])}")
    problems += [v for v in inv.cut_violations(report) if not v.startswith("cut_edge_unpaired")]
    by_edge: dict[frozenset, list] = {}
    for t in tris:
        for p, q in inv.tri_edges(t):
            by_edge.setdefault(frozenset((inv.pkey(p), inv.pkey(q))), []).append(t)
    bad = 0
    for a, b in mesh.cut_edges:
        p, q = inv.pkey((a.x, a.y)), inv.pkey((b.x, b.y))
        sides = sorted(1 if inv.cross(p, q, inv.centroid(t)) > 0 else -1 for t in by_edge.get(frozenset((p, q)), []))
        bad += sides != [-1, 1]
    if bad:
        problems.append(f"cut edges without a triangle on each side: {bad}")
    return problems, metrics(mesh)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--mode", nargs="*", default=["fast", "balanced", "accurate"])
    args = parser.parse_args()
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication(["conformity"])  # noqa: F841
    for name in args.only or perf_reference.CASES:
        for mode in args.mode:
            problems, m = check(name, TriangulationMode(mode))
            status = "OK" if not problems else "; ".join(problems)
            print(f"{name:28} {mode:9} minAng={m['min_angle']:6.2f} min/med={m['min_over_median']:.3f} "
                  f"<10°={m['angles_lt10']:3}  {status}", flush=True)


if __name__ == "__main__":
    main()
