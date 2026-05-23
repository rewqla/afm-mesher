from dataclasses import dataclass
from math import isfinite

from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle


@dataclass(slots=True)
class Mesh:
    triangles: list[Triangle]
    meters_per_pixel: float = 1.0

    def __post_init__(self) -> None:
        if not self.triangles:
            raise ValueError("Mesh triangles must not be empty.")
        if not _is_positive_finite_number(self.meters_per_pixel):
            raise ValueError("Mesh meters_per_pixel must be a positive finite number.")

    def linear_triangles(self, key_precision: int = 10, meters_per_pixel: float | None = None) -> list[LinearTriangle]:
        if key_precision < 0:
            raise ValueError("key_precision must be >= 0")
        resolved_meters_per_pixel = self.meters_per_pixel if meters_per_pixel is None else meters_per_pixel
        if not _is_positive_finite_number(resolved_meters_per_pixel):
            raise ValueError("meters_per_pixel must be a positive finite number.")

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
                    meters_per_pixel=float(resolved_meters_per_pixel),
                )
            )
        return linear


def _is_positive_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(float(value)) and float(value) > 0.0
