import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from PySide6.QtWidgets import QApplication, QMessageBox

from src.domain.entities.mesh import Mesh, TriangulationTimings
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.presentation.paint_app import PaintApp
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter


class TestAdapterTimings(unittest.TestCase):
    def test_adapter_returns_non_negative_stage_timings_within_total(self) -> None:
        outer = [(10, 10), (210, 10), (210, 160), (10, 160), (10, 10)]
        cut = [(110, 10), (110, 100)]

        mesh, _ = TriangulationAdapter().run_from_contours([outer, cut], mode=TriangulationMode.FAST)

        timings = mesh.timings
        self.assertIsNotNone(timings)
        stages = (timings.contours_s, timings.afm_s, timings.postprocess_s, timings.numbering_s)
        self.assertTrue(all(seconds >= 0.0 for seconds in stages))
        self.assertGreater(timings.afm_s, 0.0)
        self.assertGreater(timings.numbering_s, 0.0)
        self.assertLessEqual(sum(stages), timings.total_s + 1e-3)


class TestPaintAppTimingMessage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _mesh(self, timings: TriangulationTimings | None) -> Mesh:
        return Mesh(
            triangles=[Triangle(a=Point(0.0, 0.0), b=Point(1.0, 0.0), c=Point(0.0, 1.0))],
            timings=timings,
        )

    def test_time_follows_quality_and_bandwidth_in_status_message_and_failure_shows_time_to_error(self) -> None:
        window = PaintApp()
        timings = TriangulationTimings(total_s=1.1, contours_s=0.05, afm_s=0.9, postprocess_s=0.1, numbering_s=0.04)
        window._triangulation_started_at = 100.0  # noqa: SLF001
        with patch("src.presentation.paint_app.perf_counter", return_value=101.234):
            window._on_triangulation_finished(self._mesh(timings), 0.5)  # noqa: SLF001

        message = window.statusBar().currentMessage()
        self.assertIn("коефіцієнт=0.5000", message)
        self.assertTrue(
            message.endswith("; Час: 1.23 с (контури 0.05 · AFM 0.90 · постобробка 0.10 · нумерація 0.04)"),
            message,
        )

        window._triangulation_started_at = 100.0  # noqa: SLF001
        with patch("src.presentation.paint_app.perf_counter", return_value=100.25), patch.object(
            QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok
        ):
            window._on_triangulation_failed("boom")  # noqa: SLF001

        self.assertEqual(window.statusBar().currentMessage(), "Тріангуляція не вдалась; Час до помилки: 250 мс")

    def test_time_uses_milliseconds_and_english_locale(self) -> None:
        window = PaintApp()
        window._language_combo.setCurrentIndex(1)  # noqa: SLF001
        timings = TriangulationTimings(total_s=0.4, contours_s=0.01, afm_s=0.3, postprocess_s=0.05, numbering_s=0.02)
        window._triangulation_started_at = 10.0  # noqa: SLF001
        with patch("src.presentation.paint_app.perf_counter", return_value=10.42):
            window._on_triangulation_finished(self._mesh(timings), 0.5)  # noqa: SLF001

        self.assertTrue(
            window.statusBar().currentMessage().endswith(
                "; Time: 420 ms (contours 10 · AFM 300 · postprocessing 50 · numbering 20)"
            ),
            window.statusBar().currentMessage(),
        )

    def test_result_without_measured_run_has_no_time_suffix(self) -> None:
        window = PaintApp()
        window.display_triangulation_result(self._mesh(None), 0.5)
        self.assertNotIn("Час", window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
