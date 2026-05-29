import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from src.application.services.boundary_validator import BoundaryValidator
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.application.services.region_topology import RegionPolygon
from src.presentation.tools import Tool
from src.presentation.paint_app import PaintApp
from src.presentation.triangulation_adapter import TriangulationAdapter


class TestLoadedImageValidationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.adapter = TriangulationAdapter()
        self.validator = BoundaryValidator()

    def test_open_loaded_stroke_is_reported_as_open_contour(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        contours = self.adapter.extract_stroke_contours_from_image(image)
        self.assertGreaterEqual(len(contours), 1)
        self.assertNotEqual(contours, [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]])

    def test_closed_loaded_loop_is_reported_as_valid(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawEllipse(20, 20, 80, 80)
        painter.end()

        contours = self.adapter.extract_stroke_contours_from_image(image)
        result = self.validator.validate(contours)

        self.assertGreaterEqual(len(contours), 1)
        self.assertTrue(result.is_valid)

    def test_branching_loaded_stroke_reports_branch_points(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawLine(15, 60, 105, 60)
        painter.drawLine(60, 15, 60, 105)
        painter.end()

        branch_points = self.adapter.detect_stroke_branch_points_from_image(image)

        self.assertGreaterEqual(len(branch_points), 1)

    def test_paint_app_syncs_loaded_image_contours_into_canvas_geometry(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        contours = app._refresh_geometry_from_canvas_image(image, prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(len(contours), 1)
        self.assertEqual(contours, app._canvas.geometry_contours())  # noqa: SLF001

    def test_paint_app_refreshes_dirty_geometry_cache_before_triangulation(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        app._canvas.set_geometry_contours(  # noqa: SLF001
            [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        app._canvas._geometry_dirty = True  # noqa: SLF001

        contours = app._contours_for_triangulation(image, prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(len(contours), 1)
        self.assertEqual(contours, [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]])

    def test_paint_app_prefers_boundary_contours_for_multiple_closed_regions(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(10, 10, 90, 90)
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawRect(30, 30, 12, 12)
        painter.drawRect(60, 30, 12, 12)
        painter.end()

        app = PaintApp()
        app._canvas.set_image(image, record_history=False)  # noqa: SLF001
        app._canvas._geometry_dirty = True  # noqa: SLF001

        contours = app._contours_for_triangulation(image, prefer_strokes=True)  # noqa: SLF001
        closed = self.validator.normalize_valid_closed_contours(contours)

        self.assertGreaterEqual(len(closed), 3)

    def test_paint_app_preserves_open_contours_for_triangulation(self) -> None:
        app = PaintApp()
        contours = [
            [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 0.0)],
            [(5.0, 15.0), (25.0, 15.0)],
        ]

        prepared = app._prepare_contours_for_triangulation(contours)  # noqa: SLF001

        self.assertEqual(len(prepared), 2)
        self.assertIn([(5.0, 15.0), (25.0, 15.0)], prepared)
        self.assertTrue(any(len(contour) >= 4 for contour in prepared))

    def test_paint_app_keeps_single_closed_contour_alongside_open_line(self) -> None:
        app = PaintApp()
        app._canvas.set_image(self._blank_image(), record_history=False)  # noqa: SLF001
        app._canvas._geometry_dirty = True  # noqa: SLF001

        with patch.object(
            TriangulationAdapter,
            "extract_contours_from_image",
            return_value=[
                [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 0.0)],
                [(5.0, 15.0), (25.0, 15.0)],
            ],
        ):
            contours = app._contours_for_triangulation(self._blank_image(), prefer_strokes=True)  # noqa: SLF001

        self.assertGreaterEqual(len(contours), 1)
        self.assertTrue(any(len(contour) >= 4 for contour in contours))

    def test_paint_app_keeps_cached_closed_contour_when_image_extractor_returns_only_open_geometry(self) -> None:
        app = PaintApp()
        app._canvas.set_geometry_contours(  # noqa: SLF001
            [[(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 0.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        app._canvas._geometry_dirty = True  # noqa: SLF001

        with patch.object(
            TriangulationAdapter,
            "extract_contours_from_image",
            return_value=[[(5.0, 15.0), (25.0, 15.0)]],
        ):
            contours = app._contours_for_triangulation(self._blank_image(), prefer_strokes=True)  # noqa: SLF001

        self.assertEqual(len(contours), 1)
        self.assertTrue(any(len(contour) >= 4 for contour in contours))

    def test_paint_app_recovers_closed_boundary_from_image_when_cache_has_only_open_contours(self) -> None:
        app = PaintApp()
        app._canvas.set_geometry_contours(  # noqa: SLF001
            [[(5.0, 15.0), (25.0, 15.0)]],
            preprocess=False,
            redraw_image=False,
            emit_change=False,
        )
        app._canvas._geometry_dirty = True  # noqa: SLF001

        with patch.object(
            TriangulationAdapter,
            "extract_contours_from_image",
            return_value=[
                [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 0.0)],
                [(5.0, 15.0), (25.0, 15.0)],
            ],
        ):
            contours = app._contours_for_triangulation(self._blank_image(), prefer_strokes=True)  # noqa: SLF001

        self.assertEqual(len(contours), 2)
        self.assertTrue(any(len(contour) >= 4 for contour in contours))
        self.assertTrue(any(len(contour) == 2 for contour in contours))

    def test_paint_app_detects_almost_closed_contour_as_closed(self) -> None:
        app = PaintApp()
        contours = [
            [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.8, 0.4)],
            [(5.0, 15.0), (25.0, 15.0)],
        ]

        closed, open_contours = app._split_closed_and_open_contours(contours)  # noqa: SLF001

        self.assertEqual(len(closed), 1)
        self.assertEqual(len(open_contours), 1)

    def test_switching_back_to_pen_resyncs_dirty_geometry_from_canvas_image(self) -> None:
        image = self._blank_image()
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 4))
        painter.drawArc(20, 20, 80, 80, 35 * 16, 280 * 16)
        painter.end()

        app = PaintApp()
        app._canvas.set_image(image, record_history=False)  # noqa: SLF001
        app._canvas._geometry_contours = []  # noqa: SLF001
        app._canvas._geometry_dirty = True  # noqa: SLF001

        app._set_tool(Tool.PEN)  # noqa: SLF001

        contours = app._canvas.geometry_contours()  # noqa: SLF001
        result = self.validator.validate(contours)
        self.assertGreaterEqual(len(contours), 1)
        self.assertFalse(app._canvas.geometry_is_dirty())  # noqa: SLF001
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.open_contours), 1)

    def test_adapter_merges_overlapping_holes_before_afm(self) -> None:
        outer = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
        hole_1 = [(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)]
        hole_2 = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            result_mesh, coefficient = self.adapter.run_from_contours([outer, hole_1, hole_2])

        self.assertEqual(result_mesh.triangles, mesh.triangles)
        self.assertGreaterEqual(coefficient, 0.0)
        self.assertEqual(generate.call_count, 1)
        call_args = generate.call_args.args
        boundary, holes, _ = call_args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(holes), 1)

    def test_adapter_keeps_separate_holes_separate_before_afm(self) -> None:
        outer = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
        hole_1 = [(5, 5), (9, 5), (9, 9), (5, 9), (5, 5)]
        hole_2 = [(18, 5), (22, 5), (22, 9), (18, 9), (18, 5)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            result_mesh, coefficient = self.adapter.run_from_contours([outer, hole_1, hole_2])

        self.assertEqual(result_mesh.triangles, mesh.triangles)
        self.assertGreaterEqual(coefficient, 0.0)
        call_args = generate.call_args.args
        boundary, holes, _ = call_args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(holes), 2)

    def test_adapter_collects_nested_inner_polygons_as_obstacles(self) -> None:
        outer = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
        hole_like = [(8, 8), (32, 8), (32, 32), (8, 32), (8, 8)]
        island_like = [(14, 14), (26, 14), (26, 26), (14, 26), (14, 14)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            result_mesh, coefficient = self.adapter.run_from_contours([outer, hole_like, island_like])

        self.assertEqual(result_mesh.triangles, mesh.triangles)
        self.assertGreaterEqual(coefficient, 0.0)
        boundary, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertGreaterEqual(len(holes), 1)

    def test_adapter_merges_touching_holes_chain_into_single_obstacle(self) -> None:
        outer = [(0, 0), (60, 0), (60, 40), (0, 40), (0, 0)]
        h1 = [(8, 10), (18, 10), (18, 20), (8, 20), (8, 10)]
        h2 = [(18, 10), (28, 10), (28, 20), (18, 20), (18, 10)]
        h3 = [(28, 10), (38, 10), (38, 20), (28, 20), (28, 10)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            self.adapter.run_from_contours([outer, h1, h2, h3])

        _, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(holes), 1)

    def test_adapter_preserves_two_separate_hole_clusters(self) -> None:
        outer = [(0, 0), (80, 0), (80, 50), (0, 50), (0, 0)]
        # cluster A (touching)
        a1 = [(8, 8), (16, 8), (16, 16), (8, 16), (8, 8)]
        a2 = [(16, 8), (24, 8), (24, 16), (16, 16), (16, 8)]
        # cluster B (overlapping)
        b1 = [(50, 20), (60, 20), (60, 30), (50, 30), (50, 20)]
        b2 = [(56, 24), (66, 24), (66, 34), (56, 34), (56, 24)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            self.adapter.run_from_contours([outer, a1, a2, b1, b2])

        _, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(holes), 2)

    def test_adapter_merges_f_like_obstacle_from_multiple_rectangles(self) -> None:
        outer = [(0, 0), (120, 0), (120, 120), (0, 120), (0, 0)]
        vertical = [(30, 20), (50, 20), (50, 100), (30, 100), (30, 20)]
        top_bar = [(30, 80), (90, 80), (90, 100), (30, 100), (30, 80)]
        mid_bar = [(30, 50), (75, 50), (75, 65), (30, 65), (30, 50)]
        left_stub = [(20, 60), (30, 60), (30, 75), (20, 75), (20, 60)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            self.adapter.run_from_contours([outer, vertical, top_bar, mid_bar, left_stub])

        _, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(holes), 1)

    def test_adapter_merges_corner_touching_holes_into_single_obstacle(self) -> None:
        outer = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
        h1 = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        h2 = [(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            self.adapter.run_from_contours([outer, h1, h2])

        _, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(holes), 1)

    def test_adapter_ignores_obstacle_outside_selected_shell(self) -> None:
        outer = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
        inner = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        outside = [(55, 55), (65, 55), (65, 65), (55, 65), (55, 55)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            self.adapter.run_from_contours([outer, inner, outside])

        boundary, holes, _ = generate.call_args.args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(holes), 1)

    def test_adapter_routes_open_segment_into_cuts(self) -> None:
        outer = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
        cut = [(5, 15), (25, 15)]

        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.return_value = mesh

            result_mesh, coefficient = self.adapter.run_from_contours([outer, cut])

        self.assertEqual(result_mesh.triangles, mesh.triangles)
        self.assertGreaterEqual(coefficient, 0.0)
        call_args = generate.call_args.args
        boundary, holes, cuts = call_args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(holes), 0)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0], [Point(5.0, 15.0), Point(25.0, 15.0)])

    def test_adapter_treats_simplified_hand_drawn_loop_without_duplicate_endpoint_as_closed(self) -> None:
        hand_drawn_loop = [(0, 0), (60, 2), (61, 42), (28, 58), (2, 40), (1, 1)]
        cut = [(10, 25), (40, 25)]

        closed, open_contours = self.adapter._split_closed_and_open_contours([hand_drawn_loop, cut], epsilon=2.0)  # noqa: SLF001

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0][0], closed[0][-1])
        self.assertEqual(len(open_contours), 1)

    def test_adapter_splits_cut_crossing_hole_into_two_segments(self) -> None:
        hole = RegionPolygon([Point(30, 30), Point(70, 30), Point(70, 70), Point(30, 70)])
        cut = [Point(10, 50), Point(90, 50)]

        parts = self.adapter._split_cut_by_hole_conflicts(cut, [hole])  # noqa: SLF001

        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0][0].x < 30.0 and parts[0][1].x < 30.0)
        self.assertTrue(parts[1][0].x > 70.0 and parts[1][1].x > 70.0)

    def test_adapter_merges_overlapping_collinear_cuts(self) -> None:
        cuts = [
            [Point(10, 30), Point(90, 30)],
            [Point(20, 30), Point(80, 30)],
        ]

        merged = self.adapter._merge_collinear_overlapping_cuts(cuts)  # noqa: SLF001

        self.assertEqual(len(merged), 1)
        seg = merged[0]
        self.assertAlmostEqual(seg[0].y, 30.0)
        self.assertAlmostEqual(seg[1].y, 30.0)
        self.assertLessEqual(min(seg[0].x, seg[1].x), 10.0)
        self.assertGreaterEqual(max(seg[0].x, seg[1].x), 90.0)

    def test_adapter_falls_back_to_pruned_cuts_after_overlap_error(self) -> None:
        outer = [(0, 0), (120, 0), (120, 120), (0, 120), (0, 0)]
        cut_1 = [(10, 30), (95, 30)]
        cut_2 = [(15, 30), (100, 30)]
        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])

        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.side_effect = [ValueError("Overlapping collinear cut segments are not supported."), mesh]
            self.adapter.run_from_contours([outer, cut_1, cut_2])

        self.assertEqual(generate.call_count, 2)
        first_cuts = generate.call_args_list[0].args[-1]
        second_cuts = generate.call_args_list[1].args[-1]
        self.assertGreaterEqual(len(first_cuts), 1)
        self.assertLessEqual(len(second_cuts), len(first_cuts))

    def test_adapter_falls_back_to_no_cuts_when_previous_attempts_fail(self) -> None:
        outer = [(0, 0), (120, 0), (120, 120), (0, 120), (0, 0)]
        cut = [(10, 40), (100, 40)]
        mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])

        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.side_effect = [
                ValueError("AFM stalled: active front cannot be advanced further."),
                ValueError("AFM stalled: active front cannot be advanced further."),
                mesh,
            ]
            self.adapter.run_from_contours([outer, cut])

        self.assertEqual(generate.call_count, 3)
        third_cuts = generate.call_args_list[2].args[-1]
        self.assertEqual(third_cuts, [])

    def _blank_image(self) -> QImage:
        image = QImage(120, 120, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        return image


if __name__ == "__main__":
    unittest.main()
