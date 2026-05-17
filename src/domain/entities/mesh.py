from dataclasses import dataclass

from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle


@dataclass(slots=True)
class Mesh:
    triangles: list[Triangle]

    def __post_init__(self) -> None:
        if not self.triangles:
            raise ValueError("Mesh triangles must not be empty.")

    def linear_triangles(self, key_precision: int = 10) -> list[LinearTriangle]:
        if key_precision < 0:
            raise ValueError("key_precision must be >= 0")

        key_to_node_number: dict[tuple[float, float], int] = {}

        def point_key(point: Point) -> tuple[float, float]:
            return (round(point.x, key_precision), round(point.y, key_precision))

        def get_node_number(point: Point) -> int:
            key = point_key(point)
            node_number = key_to_node_number.get(key)
            if node_number is not None:
                return node_number
            node_number = len(key_to_node_number) + 1
            key_to_node_number[key] = node_number
            return node_number

        linear: list[LinearTriangle] = []
        for triangle_index, triangle in enumerate(self.triangles, start=1):
            linear.append(
                LinearTriangle(
                    triangle_number=triangle_index,
                    node_coordinates=(triangle.a, triangle.b, triangle.c),
                    node_numbers=(
                        get_node_number(triangle.a),
                        get_node_number(triangle.b),
                        get_node_number(triangle.c),
                    ),
                )
            )
        return linear
