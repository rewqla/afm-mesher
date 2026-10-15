"""A: drawn / image curves are resampled to the meshing step (no tiny or needle triangles along them).
B: a loaded picture keeps filled shapes as inclusions and turns only thin pen strokes into cuts."""
from __future__ import annotations

import math
import os
import statistics
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.triangulation_adapter as adapter_module  # noqa: E402
import test_mesh_conformity_invariants as inv  # noqa: E402
from src.domain.entities.point import Point  # noqa: E402
from src.domain.geometry.resample import resample_polyline  # noqa: E402
from src.presentation.canvas import Canvas  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
UI_BUGS = ROOT / "diagnostics" / "cases" / "ui_bugs"
# Needle threshold: AFM targets near-equilateral triangles; below 15 degrees only input features
# (a cut a few px from a contour, nearly tangent contours) should force such angles.
MIN_ANGLE_DEG = 15.0
MIN_AREA_TO_MEDIAN = 0.1


def _min_angle(t) -> float:
    def angle(p, q, r) -> float:
        v1 = (q.x - p.x, q.y - p.y)
        v2 = (r.x - p.x, r.y - p.y)
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    return min(angle(t.a, t.b, t.c), angle(t.b, t.c, t.a), angle(t.c, t.a, t.b))


def _area(t) -> float:
    return abs((t.b.x - t.a.x) * (t.c.y - t.a.y) - (t.b.y - t.a.y) * (t.c.x - t.a.x)) / 2.0


def _binary(path: Path) -> QImage:
    QApplication.instance() or QApplication([])
    return Canvas.binarize_image(QImage(str(path)), threshold=127)


def _run_image_capturing_input(path: Path, mode: TriangulationMode):
    """adapter.run on a picture, plus the exact boundary / holes / cuts the mesher received."""
    captured: dict[str, tuple] = {}

    class RecordingMesher(afm_module.AdvancingFrontMesher):
        def generate_with_holes_and_cuts(self, boundary, holes, cuts, **kwargs):
            mesh = super().generate_with_holes_and_cuts(boundary, holes, cuts, **kwargs)
            captured["input"] = (list(boundary), [list(h) for h in holes], [list(c) for c in cuts])
            return mesh

    with patch.object(adapter_module, "AdvancingFrontMesher", RecordingMesher):
        mesh, _ = adapter_module.TriangulationAdapter().run(_binary(path), mode=mode)
    return mesh, captured["input"]


class TestResamplePolyline(unittest.TestCase):
    def test_dense_circle_becomes_even_steps(self) -> None:
        circle = [Point(100 + 50 * math.cos(2 * math.pi * k / 200), 100 + 50 * math.sin(2 * math.pi * k / 200))
                  for k in range(200)]
        ring = resample_polyline(circle, 20.0, closed=True)
        sides = [math.dist((p.x, p.y), (q.x, q.y)) for p, q in zip(ring, [*ring[1:], ring[0]])]
        self.assertEqual(len(ring), 16)
        self.assertLess(max(sides) - min(sides), 1e-6)

    def test_corners_and_long_sides_are_kept(self) -> None:
        # Bottom side drawn densely (1 px), others long: corners stay, the dense side is resampled.
        rect = [Point(float(x), 0.0) for x in range(0, 100)] + [Point(100, 0), Point(100, 50), Point(0, 50)]
        ring = resample_polyline(rect, 20.0, closed=True)
        for corner in ((0, 0), (100, 0), (100, 50), (0, 50)):
            self.assertIn(Point(*corner), ring)
        self.assertEqual(len(ring), 8)

    def test_anchor_on_contour_is_kept_exactly(self) -> None:
        dense = [Point(float(x), 0.0) for x in range(0, 101)]
        anchor = Point(37.5, 0.4)
        line = resample_polyline(dense, 20.0, closed=False, anchors=[(anchor, 1.0, True)])
        self.assertIn(anchor, line)
        self.assertEqual((line[0], line[-1]), (dense[0], dense[-1]))


class TestNoTinyTrianglesAlongCurves(unittest.TestCase):
    def assert_regular_near_curves(self, mesh, label: str) -> None:
        angles = [_min_angle(t) for t in mesh.triangles]
        areas = [_area(t) for t in mesh.triangles]
        self.assertGreaterEqual(min(angles), MIN_ANGLE_DEG, f"{label}: needle triangle")
        self.assertGreaterEqual(min(areas), MIN_AREA_TO_MEDIAN * statistics.median(areas), f"{label}: tiny triangle")

    def test_image_with_circular_holes(self) -> None:
        # Before resampling: min area 2 px^2 at median ~700 / ~360, 37 / 34 triangles under 10 degrees.
        for mode in (TriangulationMode.FAST, TriangulationMode.BALANCED):
            with self.subTest(mode=mode.value):
                mesh, _ = adapter_module.TriangulationAdapter().run(_binary(IMAGES / "spot_with_circle.png"), mode=mode)
                self.assert_regular_near_curves(mesh, f"spot_with_circle/{mode.value}")

    def test_image_circle_accurate(self) -> None:
        # Before: 7 triangles under 10 degrees where the pixel circle has 2 px steps.
        mesh, _ = adapter_module.TriangulationAdapter().run(_binary(IMAGES / "circle.png"), mode=TriangulationMode.ACCURATE)
        self.assert_regular_near_curves(mesh, "circle/accurate")


class TestLoadedPictureFillsAndStrokes(unittest.TestCase):
    """1.png: pen outline, a filled T, three pen lines (the diagonal crosses the other two), plus the
    screenshot's 1 px frame line on the right edge (outside the outline)."""

    def test_contours_of_loaded_picture(self) -> None:
        contours = adapter_module.TriangulationAdapter().extract_stroke_contours_from_image(_binary(UI_BUGS / "1.png"))

        def is_closed(c) -> bool:
            # Boundary contours come without the repeated first point; the adapter closes them by distance.
            return len(c) >= 3 and math.dist(c[0], c[-1]) <= 1.5

        closed = [c for c in contours if is_closed(c)]
        open_ = [c for c in contours if not is_closed(c)]
        self.assertEqual(len(closed), 2, "outline and the filled T")
        outline = max(closed, key=len)
        t_shape = min(closed, key=len)
        xs = [p[0] for p in t_shape]
        ys = [p[1] for p in t_shape]
        self.assertTrue(155 <= min(xs) <= 160 and 540 <= max(xs) <= 545, "T keeps its filled outline")
        self.assertTrue(160 <= min(ys) <= 167 and 400 <= max(ys) <= 405)
        ring = [(float(x), float(y)) for x, y in outline]
        inside = [c for c in open_ if inv.point_in_ring(tuple(map(float, c[len(c) // 2])), ring)]
        self.assertEqual(len(inside), 3, "three pen lines, the crossing diagonal kept whole")

    def test_balanced_mesh_has_inclusion_and_three_cuts_and_conforms(self) -> None:
        mesh, (boundary, holes, cuts) = _run_image_capturing_input(UI_BUGS / "1.png", TriangulationMode.BALANCED)
        self.assertEqual(len(holes), 1, "the T is an inclusion")
        tris = [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in mesh.triangles]
        self.assertFalse(any(inv.point_in_ring(inv.centroid(t), [(p.x, p.y) for p in holes[0]]) for t in tris))
        shell = [(p.x, p.y) for p in boundary]
        cut_lines = [[(p.x, p.y) for p in c] for c in cuts]
        for start, end in (((665, 119), (670, 402)), ((728, 289), (521, 538)), ((209, 488), (705, 487))):
            # Every drawn line is followed by cut pieces from end to end.
            covered = [c for c in cut_lines if all(inv.seg_dist(p, start, end) < 4.0 for p in c)]
            length = sum(math.dist(a, b) for c in covered for a, b in zip(c, c[1:]))
            self.assertGreater(length, 0.95 * math.dist(start, end), f"line {start}-{end}")
        report = inv.analyze_conformity(
            tris, shell, [[(p.x, p.y) for p in h] for h in holes], inv.CutGeometry.build(cut_lines, shell)
        )
        self.assertEqual(inv.overlap_violations(report), [])
        self.assertEqual(report["hanging"], [])
        # cut_edge_unpaired classifies sides by the nearest cut segment, which is ambiguous next to a
        # crossing of two cuts; each cut edge is checked against its own line below instead.
        self.assertEqual([v for v in inv.cut_violations(report) if not v.startswith("cut_edge_unpaired")], [])
        by_edge: dict[frozenset, list] = {}
        for t in tris:
            for p, q in inv.tri_edges(t):
                by_edge.setdefault(frozenset((inv.pkey(p), inv.pkey(q))), []).append(t)
        for a, b in mesh.cut_edges:
            p, q = inv.pkey((a.x, a.y)), inv.pkey((b.x, b.y))
            sides = sorted(
                1 if inv.cross(p, q, inv.centroid(t)) > 0 else -1 for t in by_edge.get(frozenset((p, q)), [])
            )
            self.assertEqual(sides, [-1, 1], f"cut edge {p}-{q} must have one triangle on each side")


if __name__ == "__main__":
    unittest.main()
