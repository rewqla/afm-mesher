import unittest
from math import isfinite
from unittest.mock import patch

import _bootstrap  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMessageBox

from src.application.services.boundary_validator import BoundaryValidator
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import point_in_polygon
from src.presentation.paint_app import PaintApp, TriangulationWorker
from src.presentation.triangulation_adapter import TriangulationAdapter
from src.presentation.tools import TriangulationMode


class TestE2ETriangulationFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.adapter = TriangulationAdapter()
        self.validator = BoundaryValidator()

    def test_draw_mode_valid_outer_with_hole_triangulates(self) -> None:
        outer = [(0, 0), (120, 0), (120, 120), (0, 120), (0, 0)]
        hole = [(40, 40), (80, 40), (80, 80), (40, 80), (40, 40)]

        validation = self.validator.validate([outer, hole])
        mesh, coefficient = self.adapter.run_from_contours([outer, hole], mode=TriangulationMode.BALANCED)

        self.assertTrue(validation.is_valid)
        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        self.assert_no_triangle_centroid_in_holes(mesh, [[Point(float(x), float(y)) for x, y in hole[:-1]]])

    def test_draw_mode_outer_with_open_cut_triangulates(self) -> None:
        outer = [(0, 0), (140, 0), (140, 120), (0, 120), (0, 0)]
        open_cut = [(15, 60), (125, 60)]

        validation = self.validator.validate([outer, open_cut])
        mesh, coefficient = self.adapter.run_from_contours([outer, open_cut], mode=TriangulationMode.BALANCED)

        self.assertTrue(validation.is_valid)
        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))

    def test_nested_shapes_inside_hole_do_not_break_triangulation(self) -> None:
        outer = [(0, 0), (140, 0), (140, 140), (0, 140), (0, 0)]
        hole = [(35, 35), (105, 35), (105, 105), (35, 105), (35, 35)]
        island_inside_hole = [(55, 55), (85, 55), (85, 85), (55, 85), (55, 55)]

        validation = self.validator.validate([outer, hole, island_inside_hole])
        mesh, coefficient = self.adapter.run_from_contours(
            [outer, hole, island_inside_hole],
            mode=TriangulationMode.BALANCED,
        )

        self.assertTrue(validation.is_valid)
        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        self.assert_no_triangle_centroid_in_holes(mesh, [[Point(float(x), float(y)) for x, y in hole[:-1]]])

    def test_invalid_outer_open_shell_fails_with_expected_error(self) -> None:
        invalid_outer_open = [(0, 0), (120, 0), (120, 120), (0, 120)]
        open_inner = [(40, 40), (80, 80)]

        validation = self.validator.validate([invalid_outer_open, open_inner])
        with self.assertRaises(ValueError) as exc:
            self.adapter.run_from_contours([invalid_outer_open, open_inner], mode=TriangulationMode.BALANCED)

        self.assertFalse(validation.is_valid)
        self.assert_expected_error_contains(exc.exception, "No closed contour region found")

    def test_load_mode_image_contours_triangulate(self) -> None:
        image = QImage(160, 160, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawRect(10, 10, 130, 130)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(55, 55, 35, 35)
        painter.end()

        mesh, coefficient = self.adapter.run(image, mode=TriangulationMode.BALANCED)

        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        hole = [[Point(55.0, 55.0), Point(90.0, 55.0), Point(90.0, 90.0), Point(55.0, 90.0)]]
        self.assert_no_triangle_centroid_in_holes(mesh, hole)

    def test_overlapping_collinear_cuts_do_not_crash_flow(self) -> None:
        outer = [(0, 0), (140, 0), (140, 120), (0, 120), (0, 0)]
        cut_1 = [(10, 60), (120, 60)]
        cut_2 = [(20, 60), (130, 60)]

        mesh, coefficient = self.adapter.run_from_contours([outer, cut_1, cut_2], mode=TriangulationMode.BALANCED)

        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))

    def test_multiple_holes_are_excluded_from_mesh(self) -> None:
        outer = [(0, 0), (180, 0), (180, 140), (0, 140), (0, 0)]
        hole_1 = [(25, 25), (55, 25), (55, 55), (25, 55), (25, 25)]
        hole_2 = [(110, 70), (150, 70), (150, 110), (110, 110), (110, 70)]

        mesh, coefficient = self.adapter.run_from_contours([outer, hole_1, hole_2], mode=TriangulationMode.BALANCED)

        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        self.assert_no_triangle_centroid_in_holes(
            mesh,
            [
                [Point(float(x), float(y)) for x, y in hole_1[:-1]],
                [Point(float(x), float(y)) for x, y in hole_2[:-1]],
            ],
        )

    def test_outside_closed_shape_is_ignored(self) -> None:
        outer = [(0, 0), (120, 0), (120, 120), (0, 120), (0, 0)]
        inner_hole = [(40, 40), (80, 40), (80, 80), (40, 80), (40, 40)]
        outside_shape = [(150, 150), (190, 150), (190, 190), (150, 190), (150, 150)]

        mesh, coefficient = self.adapter.run_from_contours(
            [outer, inner_hole, outside_shape],
            mode=TriangulationMode.BALANCED,
        )

        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        self.assert_no_triangle_centroid_in_holes(
            mesh,
            [[Point(float(x), float(y)) for x, y in inner_hole[:-1]]],
        )

    def test_cut_fallback_to_no_cuts_preserves_holes(self) -> None:
        outer = [(0, 0), (160, 0), (160, 130), (0, 130), (0, 0)]
        hole = [(50, 40), (110, 40), (110, 95), (50, 95), (50, 40)]
        cut_1 = [(10, 65), (145, 65)]
        cut_2 = [(20, 65), (150, 65)]

        fake_mesh = Mesh(triangles=[Triangle(Point(0, 0), Point(1, 0), Point(0, 1))])
        with patch("src.application.services.advancing_front_mesher.AdvancingFrontMesher.generate_with_holes_and_cuts") as generate:
            generate.side_effect = [
                ValueError("Overlapping collinear cut segments are not supported."),
                ValueError("Overlapping collinear cut segments are not supported."),
                fake_mesh,
            ]
            mesh, coefficient = self.adapter.run_from_contours([outer, hole, cut_1, cut_2], mode=TriangulationMode.BALANCED)

        self.assert_mesh_non_empty(mesh)
        self.assertTrue(isfinite(coefficient))
        self.assertEqual(generate.call_count, 3)
        boundary, holes, cuts = generate.call_args_list[2].args[-3:]
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(holes), 1)
        self.assertEqual(cuts, [])

    def test_triangulation_worker_emits_finished_on_success(self) -> None:
        image = self._filled_rect_image(140, 120, 10, 10, 90, 70)
        contours = [[(10, 10), (100, 10), (100, 80), (10, 80), (10, 10)]]

        worker = TriangulationWorker(
            adapter=self.adapter,
            image_data=image,
            contours=contours,
            mode=TriangulationMode.BALANCED,
            custom_settings=None,
        )

        payload: dict[str, object] = {}
        worker.finished.connect(lambda mesh, coef: payload.update({"mesh": mesh, "coef": coef}))
        worker.failed.connect(lambda msg: payload.update({"err": msg}))
        worker.run()

        self.assertIn("mesh", payload)
        self.assertNotIn("err", payload)
        self.assert_mesh_non_empty(payload["mesh"])  # type: ignore[arg-type]
        self.assertTrue(isfinite(float(payload["coef"])))

    def test_triangulation_worker_emits_failed_on_error(self) -> None:
        image = self._blank_image(140, 120)
        contours = [[(10, 10), (100, 10), (100, 80), (10, 80), (10, 10)]]

        payload: dict[str, object] = {}
        with patch.object(TriangulationAdapter, "run_from_contours", side_effect=ValueError("Synthetic triangulation failure")):
            worker = TriangulationWorker(
                adapter=self.adapter,
                image_data=image,
                contours=contours,
                mode=TriangulationMode.BALANCED,
                custom_settings=None,
            )
            worker.finished.connect(lambda mesh, coef: payload.update({"mesh": mesh, "coef": coef}))
            worker.failed.connect(lambda msg: payload.update({"err": msg}))
            worker.run()

        self.assertIn("err", payload)
        self.assertIn("Synthetic triangulation failure", str(payload["err"]))
        self.assertNotIn("mesh", payload)

    def test_paint_app_open_outer_contour_blocks_preparation(self) -> None:
        app = PaintApp()
        contours = [[(10.0, 10.0), (100.0, 10.0), (100.0, 80.0), (10.0, 80.0)]]

        info_calls: list[str] = []
        original_info = QMessageBox.information
        QMessageBox.information = lambda _p, _t, msg: info_calls.append(msg) or QMessageBox.StandardButton.Ok
        try:
            prepared = app._prepare_contours_for_triangulation(contours)  # noqa: SLF001
        finally:
            QMessageBox.information = original_info

        self.assertEqual(prepared, [])
        self.assertGreaterEqual(len(info_calls), 1)

    def assert_mesh_non_empty(self, mesh: Mesh) -> None:
        self.assertGreater(len(mesh.triangles), 0, "Expected non-empty triangulation mesh.")

    def assert_no_triangle_centroid_in_holes(self, mesh: Mesh, holes: list[list[Point]]) -> None:
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            for hole in holes:
                self.assertFalse(
                    point_in_polygon(centroid, hole, include_boundary=True),
                    "Triangle centroid lies inside a hole.",
                )

    def assert_expected_error_contains(self, exc: Exception, text: str) -> None:
        self.assertIn(text, str(exc))

    def _blank_image(self, width: int = 160, height: int = 160) -> QImage:
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor(Qt.GlobalColor.white))
        return image

    def _filled_rect_image(self, width: int, height: int, x: int, y: int, w: int, h: int) -> QImage:
        image = self._blank_image(width, height)
        painter = QPainter(image)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawRect(x, y, w, h)
        painter.end()
        return image


if __name__ == "__main__":
    unittest.main()
