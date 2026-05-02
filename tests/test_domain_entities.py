import unittest

import _bootstrap  # noqa: F401
from src.domain.entities.edge import Edge
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.polygon import Polygon
from src.domain.entities.triangle import Triangle


class TestDomainEntities(unittest.TestCase):
    def test_edge_rejects_identical_endpoints(self) -> None:
        p = Point(0.0, 0.0)
        with self.assertRaises(ValueError):
            Edge(start=p, end=p)

    def test_triangle_rejects_degenerate(self) -> None:
        with self.assertRaises(ValueError):
            Triangle(
                a=Point(0.0, 0.0),
                b=Point(1.0, 1.0),
                c=Point(2.0, 2.0),
            )

    def test_triangle_accepts_non_degenerate(self) -> None:
        triangle = Triangle(
            a=Point(0.0, 0.0),
            b=Point(1.0, 0.0),
            c=Point(0.0, 1.0),
        )
        self.assertGreater(triangle.area(), 0.0)

    def test_polygon_requires_non_empty_vertices(self) -> None:
        with self.assertRaises(ValueError):
            Polygon(vertices=[])

    def test_polygon_requires_at_least_three_vertices(self) -> None:
        with self.assertRaises(ValueError):
            Polygon(vertices=[Point(0.0, 0.0), Point(1.0, 0.0)])

    def test_mesh_requires_non_empty_triangles(self) -> None:
        with self.assertRaises(ValueError):
            Mesh(triangles=[])


if __name__ == "__main__":
    unittest.main()
