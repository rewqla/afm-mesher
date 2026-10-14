"""Зауваження 2: розріз, який бачить користувач, — це саме ланцюжок ребер сітки (mesh.cut_edges).

Сирий штрих пера адаптер спрощує (contour_epsilon), тому інваріанти перевіряються відносно
mesh.cut_edges — того, що малюють canvas / debug-візуалізація / експорт.
"""
from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

import test_mesh_conformity_invariants as inv
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.point import Point
from src.infrastructure.export.mesh_result_exporter import export_triangles
from src.presentation.canvas import Canvas
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter


def _square(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _noisy_stroke(start: tuple[float, float], end: tuple[float, float], n: int, seed: int) -> list[tuple[int, int]]:
    """Щільна полілінія з шумом ±1 px, як штрих пера на canvas (цілі пікселі)."""
    rng = random.Random(seed)
    out = []
    for i in range(n + 1):
        t = i / n
        x = start[0] + (end[0] - start[0]) * t + 15.0 * math.sin(3.0 * t)
        y = start[1] + (end[1] - start[1]) * t
        if 0 < i < n:
            x += rng.choice((-1, 0, 1))
            y += rng.choice((-1, 0, 1))
        out.append((int(round(x)), int(round(y))))
    return out


def _displayed_cut(mesh) -> list[list[tuple[float, float]]]:
    return [[(a.x, a.y), (b.x, b.y)] for a, b in mesh.cut_edges]


def _mesh_edges(mesh) -> set[frozenset]:
    return {
        frozenset((inv.pkey((p.x, p.y)), inv.pkey((q.x, q.y))))
        for t in mesh.triangles
        for p, q in ((t.a, t.b), (t.b, t.c), (t.c, t.a))
    }


class TestDisplayedCutMatchesMesh(unittest.TestCase):
    def _assert_conforming(self, mesh, shell: list[tuple[float, float]], holes=()) -> None:
        self.assertTrue(mesh.cut_edges, "сітка не повідомила ребра розрізу")
        edges = _mesh_edges(mesh)
        for a, b in mesh.cut_edges:
            self.assertIn(frozenset((inv.pkey((a.x, a.y)), inv.pkey((b.x, b.y)))), edges)
        tris = [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in mesh.triangles]
        cuts = inv.CutGeometry.build(_displayed_cut(mesh), shell)
        report = inv.analyze_conformity(tris, shell, [list(h) for h in holes], cuts)
        self.assertEqual(inv.cut_violations(report), [])
        self.assertEqual(report["hanging"], [])

    def test_handdrawn_cases_conform_to_displayed_cut(self) -> None:
        for name in ("fig22_handdrawn.json", "fig22_handdrawn_cut_from_boundary.json"):
            with self.subTest(case=name):
                case = inv.mesh_case(name, inv.FAST)
                self._assert_conforming(case.mesh, case.shell, case.holes)

    def test_raw_pen_stroke_through_adapter(self) -> None:
        stroke = _noisy_stroke((120.0, 110.0), (150.0, 300.0), n=180, seed=7)
        adapter = TriangulationAdapter()
        mesh, _ = adapter.run_from_contours([_square(50, 50, 350, 350), stroke], mode=TriangulationMode.FAST)
        shell = [(50.0, 50.0), (350.0, 50.0), (350.0, 350.0), (50.0, 350.0)]
        self._assert_conforming(mesh, shell)
        # Відображуваний розріз — спрощена ламана, а не сирий штрих (тому й малюємо cut_edges).
        self.assertLess(len(mesh.cut_edges), len(stroke) - 1)
        ends = {inv.pkey((p.x, p.y)) for a, b in mesh.cut_edges for p in (a, b)}
        self.assertIn(inv.pkey(stroke[0]), ends)
        self.assertIn(inv.pkey(stroke[-1]), ends)

    def test_export_triangles_contains_cut_edges(self) -> None:
        mesh = inv.mesh_case("fig22_handdrawn.json", inv.FAST).mesh
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triangles.json"
            export_triangles(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["cut_edges"],
            [[[a.x, a.y], [b.x, b.y]] for a, b in mesh.cut_edges],
        )


class TestCanvasDrawsMeshCut(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_canvas_draws_cut_edges_instead_of_raw_stroke(self) -> None:
        stroke = _noisy_stroke((120.0, 110.0), (150.0, 300.0), n=180, seed=7)
        mesh, _ = TriangulationAdapter().run_from_contours(
            [_square(50, 50, 350, 350), stroke], mode=TriangulationMode.FAST
        )
        canvas = Canvas(width=400, height=400)
        # Сирий штрих, зсунутий на 20 px від розрізу сітки: під сіткою він не малюється.
        shifted = [(x + 20.0, y) for x, y in stroke]
        canvas.set_geometry_contours(
            [[(float(x), float(y)) for x, y in _square(50, 50, 350, 350)], shifted],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        canvas.set_mesh_overlay(mesh)
        image = canvas.rendered_image()

        a, b = mesh.cut_edges[len(mesh.cut_edges) // 2]
        mid = image.pixelColor(int(round((a.x + b.x) / 2)), int(round((a.y + b.y) / 2)))
        self.assertEqual(mid.name(), QColor("#111827").name())

        raw_red = QColor("#dc2626").name()
        self.assertFalse(
            any(image.pixelColor(int(x), int(y)).name() == raw_red for x, y in shifted[20:-20:10]),
            "сирий штрих розрізу намальований поверх сітки",
        )


class TestCutCollinearRejection(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher()
        self.cuts = [(Point(0, 0), Point(10, 0)), (Point(10, 0), Point(20, 0))]

    def respects(self, a, b, c) -> bool:
        return self.mesher._triangle_respects_cut_segments(Point(*a), Point(*b), Point(*c), self.cuts)  # noqa: SLF001

    def test_edge_along_cut_overlapping_segment_rejected(self) -> None:
        self.assertFalse(self.respects((-5, 0), (5, 0), (0, 5)))

    def test_edge_through_cut_node_rejected(self) -> None:
        # Ребро (5,0)-(15,0) лежить на розрізі й проходить через його вузол (10,0).
        self.assertFalse(self.respects((5, 0), (15, 0), (10, 6)))

    def test_vertex_inside_cut_segment_rejected(self) -> None:
        self.assertFalse(self.respects((5, 0), (8, 5), (2, 5)))

    def test_edge_crossing_cut_exactly_at_node_rejected(self) -> None:
        self.assertFalse(self.respects((10, -4), (10, 4), (4, 0.5)))

    def test_cut_running_into_triangle_from_vertex_rejected(self) -> None:
        self.assertFalse(self.respects((10, 0), (25, -5), (25, 5)))

    def test_triangle_on_cut_edge_accepted(self) -> None:
        self.assertTrue(self.respects((0, 0), (10, 0), (5, 8)))
        self.assertTrue(self.respects((10, 0), (0, 0), (5, -8)))

    def test_triangle_touching_cut_end_from_outside_accepted(self) -> None:
        self.assertTrue(self.respects((20, 0), (25, 5), (25, -5)))


if __name__ == "__main__":
    unittest.main()
