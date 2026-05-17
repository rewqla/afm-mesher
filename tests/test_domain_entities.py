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

    def test_triangle_orientation_sign(self) -> None:
        ccw = Triangle(
            a=Point(0.0, 0.0),
            b=Point(1.0, 0.0),
            c=Point(0.0, 1.0),
        )
        cw = Triangle(
            a=Point(0.0, 0.0),
            b=Point(0.0, 1.0),
            c=Point(1.0, 0.0),
        )
        self.assertEqual(ccw.orientation(), 1)
        self.assertEqual(cw.orientation(), -1)

    def test_triangle_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Triangle(
                a=Point(float("inf"), 0.0),
                b=Point(1.0, 0.0),
                c=Point(0.0, 1.0),
            )

    def test_triangle_rejects_duplicated_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Triangle(
                a=Point(0.0, 0.0),
                b=Point(0.0, 0.0),
                c=Point(0.0, 1.0),
            )

    def test_polygon_requires_non_empty_vertices(self) -> None:
        with self.assertRaises(ValueError):
            Polygon(vertices=[])

    def test_polygon_requires_at_least_three_vertices(self) -> None:
        with self.assertRaises(ValueError):
            Polygon(vertices=[Point(0.0, 0.0), Point(1.0, 0.0)])

    def test_mesh_requires_non_empty_triangles(self) -> None:
        with self.assertRaises(ValueError):
            Mesh(triangles=[])

    def test_mesh_builds_linear_triangles_with_global_numbering(self) -> None:
        mesh = Mesh(
            triangles=[
                Triangle(
                    a=Point(0.0, 0.0),
                    b=Point(1.0, 0.0),
                    c=Point(0.0, 1.0),
                ),
                Triangle(
                    a=Point(1.0, 0.0),
                    b=Point(1.0, 1.0),
                    c=Point(0.0, 1.0),
                ),
            ]
        )

        linear = mesh.linear_triangles()
        self.assertEqual(len(linear), 2)
        self.assertEqual(linear[0].triangle_number, 1)
        self.assertEqual(linear[1].triangle_number, 2)

        self.assertEqual(linear[0].node_numbers[1], linear[1].node_numbers[0])
        self.assertEqual(linear[0].node_numbers[2], linear[1].node_numbers[2])
        self.assertEqual(sorted({n for t in linear for n in t.node_numbers}), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
