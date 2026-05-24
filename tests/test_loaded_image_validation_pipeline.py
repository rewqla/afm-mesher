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

    def _blank_image(self) -> QImage:
        image = QImage(120, 120, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        return image


if __name__ == "__main__":
    unittest.main()
