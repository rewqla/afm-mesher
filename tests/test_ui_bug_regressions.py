"""Regressions for geometry reported from manual UI testing (diagnostics/cases/ui_bugs/*.png)."""
import json
import unittest
from math import hypot
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from PySide6.QtWidgets import QApplication, QMessageBox

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter
from src.presentation.paint_app import PaintApp
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter

ROOT = Path(__file__).resolve().parents[1]
UI_BUGS = ROOT / "diagnostics" / "cases" / "ui_bugs"


def _drawn(name: str) -> list[list[list[float]]]:
    return json.loads((UI_BUGS / name).read_text(encoding="utf-8"))["contours"]


def _key(point: Point) -> tuple[float, float]:
    return (round(point.x, 6), round(point.y, 6))


def _mesh_edges(mesh: Mesh) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    edges = set()
    for t in mesh.triangles:
        for a, b in ((t.a, t.b), (t.b, t.c), (t.c, t.a)):
            edges.add(tuple(sorted((_key(a), _key(b)))))
    return edges


def _edges_crossing_segment(mesh: Mesh, start: tuple[float, float], end: tuple[float, float]) -> int:
    """Mesh edges properly crossing the drawn cut (a cut must be followed by mesh edges, not crossed).
    Endpoints within 1e-4 of the other segment count as touching: cut nodes lie on it up to rounding."""

    def signed_distance(p, q, r) -> float:  # noqa: ANN001
        length = hypot(q[0] - p[0], q[1] - p[1])
        return ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) / length

    count = 0
    for a, b in _mesh_edges(mesh):
        d1, d2 = signed_distance(start, end, a), signed_distance(start, end, b)
        d3, d4 = signed_distance(a, b, start), signed_distance(a, b, end)
        if min(abs(d1), abs(d2), abs(d3), abs(d4)) > 1e-4 and d1 * d2 < 0 and d3 * d4 < 0:
            count += 1
    return count


class TestBug1CrossingCuts(unittest.TestCase):
    """1.png: T-shaped hole of two touching rectangles, three cuts, the diagonal one crossing the others.
    It used to stall every AFM pass at the cut crossings and run into the 120 s time budget."""

    def test_last_subdivided_piece_ends_exactly_at_non_integer_endpoint(self) -> None:
        mesher = AdvancingFrontMesher()
        crossing = Point(669.399164213, 359.490861027)
        pieces = mesher._subdivide_cut_segments([(Point(666.0, 119.0), crossing)], h=30.0)  # noqa: SLF001
        self.assertIs(pieces[-1][1], crossing)

    def test_crossing_cuts_share_one_junction_point(self) -> None:
        mesher = AdvancingFrontMesher()
        boundary = [Point(100, 50), Point(800, 50), Point(800, 600), Point(100, 600)]
        cuts = [
            [Point(666, 119), Point(670, 402)],
            [Point(728, 289), Point(521, 538)],
            [Point(208, 489), Point(706, 487)],
        ]
        segments = mesher._prepare_cuts(cuts, boundary, [], tolerance=1e-6)  # noqa: SLF001
        ends = [p for segment in segments for p in segment]
        for approx in ((669.4, 359.5), (562.9, 487.6)):
            near = {(p.x, p.y) for p in ends if hypot(p.x - approx[0], p.y - approx[1]) < 0.5}
            self.assertEqual(len(near), 1, f"junction near {approx} split into {near}")
            self.assertEqual(sum(1 for p in ends if (p.x, p.y) in near), 4)

    def test_drawn_figure_meshes_in_balanced_with_all_cuts_and_sector_twins(self) -> None:
        adapter = TriangulationAdapter(time_budget_s=60.0)
        mesh, _ = adapter.run_from_contours(_drawn("bug1_drawn_contours.json"), mode=TriangulationMode.BALANCED)

        self.assertEqual(mesh.warnings, ())
        for start, end in (((666, 119), (670, 402)), ((728, 289), (521, 538)), ((208, 489), (706, 487))):
            self.assertEqual(_edges_crossing_segment(mesh, start, end), 0)
        arms: dict[tuple[float, float], int] = {}
        for a, b in mesh.cut_edges:
            for p in (a, b):
                arms[_key(p)] = arms.get(_key(p), 0) + 1
        junctions = [k for k, n in arms.items() if n == 4]
        self.assertEqual(len(junctions), 2)
        indexed = mesh.indexed_mesh()
        for junction in junctions:
            numbers = [
                i for i, node in enumerate(indexed.nodes, start=indexed.index_base)
                if (round(node[0], 6), round(node[1], 6)) == junction
            ]
            self.assertEqual(len(numbers), 4, "one node number per sector around a cut crossing")


class TestBug3CutTouchingHole(unittest.TestCase):
    """3.png: rectangle + overlapping circles merged into one hole, a cut touching that hole and an
    independent cut. The touching cut used to fail validation and the adapter fell back to meshing
    without any cut, silently."""

    def test_mesher_accepts_cut_ending_on_hole_edge_and_rejects_crossing(self) -> None:
        mesher = AdvancingFrontMesher()
        boundary = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        hole = [Point(30, 30), Point(70, 30), Point(70, 70), Point(30, 70)]
        self.assertIsNone(mesher.cut_rejection_reason(boundary, [hole], [Point(50, 70), Point(50, 90)]))
        self.assertIsNone(mesher.cut_rejection_reason(boundary, [hole], [Point(30, 70), Point(10, 90)]))
        self.assertIsNotNone(mesher.cut_rejection_reason(boundary, [hole], [Point(50, 60), Point(50, 90)]))
        self.assertIsNotNone(mesher.cut_rejection_reason(boundary, [hole], [Point(30, 40), Point(30, 90)]))

    def test_cut_touching_hole_and_independent_cut_are_both_meshed(self) -> None:
        for case in ("bug3_cut_on_hole_edge.json", "bug3_cut_at_hole_corner.json", "bug3_cut_into_hole.json"):
            with self.subTest(case=case):
                contours = _drawn(case)
                mesh, _ = TriangulationAdapter(time_budget_s=60.0).run_from_contours(
                    contours, mode=TriangulationMode.BALANCED
                )
                self.assertEqual(mesh.warnings, ())
                cut_points = {_key(p) for edge in mesh.cut_edges for p in edge}
                self.assertIn((172.0, 368.0), cut_points)
                self.assertIn((571.0, 502.0), cut_points)
                self.assertEqual(_edges_crossing_segment(mesh, (285, 479), (571, 502)), 0)
                (x0, y0), (x1, y1) = contours[4]
                self.assertEqual(_edges_crossing_segment(mesh, ((x0 + x1) / 2, (y0 + y1) / 2), (x1, y1)), 0)

    def test_cut_inside_hole_is_reported_and_independent_cut_is_kept(self) -> None:
        contours = _drawn("bug3_drawn_contours.json")
        contours = contours[:4] + [[[230, 160], [330, 200]], contours[5]]
        mesh, _ = TriangulationAdapter(time_budget_s=60.0).run_from_contours(contours, mode=TriangulationMode.BALANCED)

        self.assertEqual(len(mesh.warnings), 1)
        self.assertIn("усередині дірки", mesh.warnings[0])
        self.assertIn((571.0, 502.0), {_key(p) for edge in mesh.cut_edges for p in edge})

    def test_fallback_without_cuts_is_reported(self) -> None:
        outer = [(10, 10), (210, 10), (210, 160), (10, 160), (10, 10)]
        cut = [(110, 40), (110, 120)]
        original = AdvancingFrontMesher.generate_with_holes_and_cuts

        def fail_with_cuts(mesher, boundary, holes, cuts, *args, **kwargs):  # noqa: ANN001
            if cuts:
                raise ValueError("synthetic failure")
            return original(mesher, boundary, holes, cuts, *args, **kwargs)

        with patch.object(AdvancingFrontMesher, "generate_with_holes_and_cuts", fail_with_cuts):
            mesh, _ = TriangulationAdapter().run_from_contours([outer, cut], mode=TriangulationMode.FAST)

        self.assertEqual(mesh.cut_edges, ())
        self.assertEqual(len(mesh.warnings), 1)
        self.assertIn("без розрізів", mesh.warnings[0])
        self.assertIn("synthetic failure", mesh.warnings[0])


class TestTriangulationWarningsInUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_warnings_are_shown_to_the_user_after_triangulation(self) -> None:
        window = PaintApp()
        mesh = Mesh(
            triangles=[Triangle(a=Point(0.0, 0.0), b=Point(1.0, 0.0), c=Point(0.0, 1.0))],
            warnings=("Розріз (1, 2)–(3, 4) не враховано: проходить через дірку.",),
        )
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok) as warning:
            window._on_triangulation_finished(mesh, 0.5)  # noqa: SLF001

        warning.assert_called_once()
        self.assertIn("проходить через дірку", warning.call_args.args[2])

    def test_no_dialog_without_warnings(self) -> None:
        window = PaintApp()
        mesh = Mesh(triangles=[Triangle(a=Point(0.0, 0.0), b=Point(1.0, 0.0), c=Point(0.0, 1.0))])
        with patch.object(QMessageBox, "warning") as warning:
            window._on_triangulation_finished(mesh, 0.5)  # noqa: SLF001
        warning.assert_not_called()


class TestBug4NestedInclusionsRendering(unittest.TestCase):
    """4.png: the "black" edges between the lower holes are pure red (255, 0, 0) in the screenshot;
    they only look dark when the image is scaled down. The only dark strokes are the cut edges."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_only_cut_edges_are_drawn_dark(self) -> None:
        window = PaintApp()
        importer = CoordinateJsonImporter()
        imported = importer.load(str(ROOT / "tests" / "fixtures" / "complex_nested_inclusions.json"))
        contours = window._fit_contours_to_canvas(importer.to_canvas_contours(imported))  # noqa: SLF001
        window._canvas.set_geometry_contours(contours, preprocess=False, redraw_image=False, emit_change=True)  # noqa: SLF001
        window._source_contours_for_triangulation = window._mesh_source_contours_from_imported(imported)  # noqa: SLF001
        mesh, _ = window.run_triangulation(window._canvas.image_data())  # noqa: SLF001
        window._canvas.set_mesh_overlay(mesh)  # noqa: SLF001
        image = window._canvas.rendered_image()  # noqa: SLF001

        def near_cut(x: int, y: int) -> bool:
            for a, b in mesh.cut_edges:
                dx, dy = b.x - a.x, b.y - a.y
                length_sq = dx * dx + dy * dy
                t = max(0.0, min(1.0, ((x - a.x) * dx + (y - a.y) * dy) / length_sq)) if length_sq else 0.0
                if hypot(x - (a.x + t * dx), y - (a.y + t * dy)) < 3.0:
                    return True
            return False

        stray = []
        for x in range(0, image.width(), 1):
            for y in range(0, image.height(), 1):
                color = image.pixelColor(x, y)
                if max(color.red(), color.green(), color.blue()) < 100 and not near_cut(x, y):
                    stray.append((x, y))
        self.assertEqual(stray, [])


if __name__ == "__main__":
    unittest.main()
