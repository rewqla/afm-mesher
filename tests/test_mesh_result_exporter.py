import json
from pathlib import Path
import tempfile
import unittest
from importlib.util import find_spec

import _bootstrap  # noqa: F401
from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.mesh import Mesh, MeshSourceContours
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter
from src.domain.geometry.geometry_utils import mesh_average_quality, mesh_quality_report
from src.infrastructure.export.mesh_result_exporter import (
    export_coordinates,
    export_nodes,
    export_statistics,
    export_triangles,
)


class TestMeshResultExporter(unittest.TestCase):
    def test_export_coordinates_round_trips_imported_coordinate_json(self) -> None:
        if find_spec("PySide6") is None or find_spec("PIL") is None or find_spec("numpy") is None:
            self.skipTest("Round-trip triangulation test requires PySide6, Pillow, and numpy.")
        from src.presentation.tools import TriangulationMode
        from src.presentation.triangulation_adapter import TriangulationAdapter

        input_payload = {
            "outer_boundary": [[0, 0], [120, 0], [120, 120], [0, 120], [0, 0]],
            "inclusions": [
                {
                    "vertices": [[40, 40], [80, 40], [80, 80], [40, 80], [40, 40]],
                    "type": "ignore",
                }
            ],
            "cuts": [
                [[15, 60], [105, 60]],
            ],
        }
        importer = CoordinateJsonImporter()
        imported = importer.loads(json.dumps(input_payload))
        source_contours = self._source_contours_from_imported(imported)
        adapter = TriangulationAdapter()
        mesh, _coefficient = adapter.run_from_contours(
            importer.to_canvas_contours(imported),
            mode=TriangulationMode.BALANCED,
            source_contours=source_contours,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coordinates.json"
            export_coordinates(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload, input_payload)

    def test_export_coordinates_without_source_contours_raises_clear_error(self) -> None:
        mesh = self._sample_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coordinates.json"
            with self.assertRaisesRegex(ValueError, "імпорт координат"):
                export_coordinates(mesh, path)

    def test_export_triangles_writes_linear_triangle_json(self) -> None:
        mesh = self._sample_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triangles.json"
            export_triangles(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["meters_per_pixel"], 0.5)
        self.assertEqual(len(payload["triangles"]), 3)
        first = payload["triangles"][0]
        self.assertEqual(first["triangle_number"], 1)
        self.assertEqual(first["degree"], "linear")
        self.assertEqual(len(first["node_numbers"]), 3)
        self.assertEqual(len(first["node_coordinates"]), 3)
        self.assertIn("area", first)

    def test_export_nodes_serializes_interface_as_cut(self) -> None:
        indexed = self._sample_indexed_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodes.json"
            export_nodes(indexed, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        attributes = {node["node_number"]: node["attribute"] for node in payload["nodes"]}
        self.assertEqual(attributes[1], "boundary")
        self.assertEqual(attributes[2], "cut")
        self.assertEqual(attributes[5], "interior")

    def test_export_statistics_writes_mesh_summary(self) -> None:
        mesh = self._sample_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            export_statistics(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["triangle_count"], 3)
        self.assertEqual(payload["node_count"], 5)
        self.assertEqual(payload["bandwidth"], 4)
        self.assertEqual(sum(payload["quality_histogram"]["counts"]), 3)
        self.assertEqual(payload["quality"]["min"], round(payload["quality"]["min"], 2))
        self.assertEqual(payload["quality"]["mean"], round(payload["quality"]["mean"], 2))
        self.assertEqual(payload["quality"]["max"], round(payload["quality"]["max"], 2))
        self.assertTrue(
            all(value == round(value, 2) for value in payload["quality_histogram"]["bins"])
        )
        self.assertTrue(
            all(value == round(value, 2) for value in payload["quality_histogram"]["percentages"])
        )

    def test_export_statistics_quality_percentages_sum_to_100(self) -> None:
        mesh = self._sample_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            export_statistics(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(sum(payload["quality_histogram"]["percentages"]), 100.0, delta=0.5)

    def test_export_statistics_mean_matches_existing_quality_helpers(self) -> None:
        mesh = self._sample_mesh()
        expected_average = mesh_average_quality(mesh)
        expected_report = mesh_quality_report(mesh)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            export_statistics(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(expected_average, expected_report.mean_quality, places=12)
        self.assertEqual(payload["quality"]["mean"], round(expected_average, 2))

    def test_export_statistics_empty_mesh_does_not_raise(self) -> None:
        mesh = Mesh.__new__(Mesh)
        mesh.triangles = []
        mesh.meters_per_pixel = 1.0
        mesh.node_order = None
        mesh.indexed_mesh_data = IndexedMesh(
            nodes=[],
            triangles=[],
            boundary_nodes=set(),
            interface_nodes=set(),
            bandwidth=0,
            index_base=1,
        )
        mesh.source_contours = None
        mesh._linear_triangles_cache = {}  # noqa: SLF001

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            export_statistics(mesh, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["triangle_count"], 0)
        self.assertEqual(payload["node_count"], 0)
        self.assertEqual(payload["bandwidth"], 0)
        self.assertEqual(payload["quality"], {"min": 0.0, "mean": 0.0, "max": 0.0})
        self.assertTrue(all(value == 0.0 for value in payload["quality_histogram"]["percentages"]))

    def _sample_mesh(self) -> Mesh:
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(1.0, 1.0),
            Point(0.0, 1.0),
            Point(0.5, 0.5),
        ]
        indexed = self._sample_indexed_mesh()
        mesh = Mesh(
            triangles=[
                Triangle(points[0], points[1], points[4]),
                Triangle(points[1], points[2], points[4]),
                Triangle(points[2], points[3], points[4]),
            ],
            meters_per_pixel=0.5,
            node_order=tuple(points),
            indexed_mesh_data=indexed,
        )
        mesh._linear_triangles_cache[(10, 0.5)] = (  # noqa: SLF001
            LinearTriangle(1, (points[0], points[1], points[4]), (1, 2, 5), meters_per_pixel=0.5),
            LinearTriangle(2, (points[1], points[2], points[4]), (2, 3, 5), meters_per_pixel=0.5),
            LinearTriangle(3, (points[2], points[3], points[4]), (3, 4, 5), meters_per_pixel=0.5),
        )
        return mesh

    def _sample_indexed_mesh(self) -> IndexedMesh:
        return IndexedMesh(
            nodes=[
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.5),
            ],
            triangles=[
                (1, 2, 5),
                (2, 3, 5),
                (3, 4, 5),
            ],
            boundary_nodes={1, 2, 3, 4},
            interface_nodes={2},
            bandwidth=4,
            index_base=1,
            meters_per_pixel=0.5,
        )

    def _source_contours_from_imported(self, imported) -> MeshSourceContours:
        return MeshSourceContours(
            outer_boundary=[(float(x), float(y)) for x, y in imported.outer_boundary],
            inclusions=[
                (
                    [(float(x), float(y)) for x, y in vertices],
                    inclusion_type,
                )
                for vertices, inclusion_type in imported.inclusions
            ],
            cuts=[
                [(float(x), float(y)) for x, y in cut]
                for cut in imported.cuts
            ],
        )


if __name__ == "__main__":
    unittest.main()
