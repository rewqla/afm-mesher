from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.infrastructure.export.mesh_exporter import export_mesh_obj, export_mesh_ply, export_mesh_vtk


class TestMeshExporter(unittest.TestCase):
    def test_export_obj_ply_vtk(self) -> None:
        mesh = self._sample_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obj_path = export_mesh_obj(mesh, tmp_path / "mesh.obj")
            ply_path = export_mesh_ply(mesh, tmp_path / "mesh.ply")
            vtk_path = export_mesh_vtk(mesh, tmp_path / "mesh.vtk")

            self.assertTrue(obj_path.exists())
            self.assertTrue(ply_path.exists())
            self.assertTrue(vtk_path.exists())
            self.assertGreater(obj_path.stat().st_size, 0)
            self.assertGreater(ply_path.stat().st_size, 0)
            self.assertGreater(vtk_path.stat().st_size, 0)

            obj_text = obj_path.read_text(encoding="utf-8")
            self.assertIn("v 0.0000000000 0.0000000000 0.0", obj_text)
            self.assertIn("f 1 2 3", obj_text)

            ply_text = ply_path.read_text(encoding="utf-8")
            self.assertIn("ply", ply_text)
            self.assertIn("format ascii 1.0", ply_text)
            self.assertIn("element vertex 4", ply_text)
            self.assertIn("element face 2", ply_text)

            vtk_text = vtk_path.read_text(encoding="utf-8")
            self.assertIn("# vtk DataFile Version 3.0", vtk_text)
            self.assertIn("POINTS 4 float", vtk_text)
            self.assertIn("POLYGONS 2 8", vtk_text)

    def _sample_mesh(self) -> Mesh:
        p00 = Point(0.0, 0.0)
        p10 = Point(1.0, 0.0)
        p11 = Point(1.0, 1.0)
        p01 = Point(0.0, 1.0)
        return Mesh(
            triangles=[
                Triangle(p00, p10, p11),
                Triangle(p00, p11, p01),
            ]
        )


if __name__ == "__main__":
    unittest.main()
