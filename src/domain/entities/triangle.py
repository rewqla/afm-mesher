from dataclasses import dataclass
from math import isclose, isfinite

from src.domain.entities.point import Point

_AREA_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Triangle:
    a: Point
    b: Point
    c: Point

    def __post_init__(self) -> None:
        self._validate_geometry()
        if self.is_degenerate():
            raise ValueError("Triangle is degenerate (zero area).")

    def area(self) -> float:
        return abs(self.signed_area())

    def signed_area(self) -> float:
        return (
            self.a.x * (self.b.y - self.c.y) +
            self.b.x * (self.c.y - self.a.y) +
            self.c.x * (self.a.y - self.b.y)
        ) / 2.0

    def orientation(self) -> int:
        signed = self.signed_area()
        if isclose(signed, 0.0, abs_tol=_AREA_EPSILON):
            return 0
        return 1 if signed > 0 else -1

    def is_degenerate(self) -> bool:
        return isclose(abs(self.signed_area()), 0.0, abs_tol=_AREA_EPSILON)

    @property
    def coordinates(self) -> tuple[Point, Point, Point]:
        return (self.a, self.b, self.c)

    @staticmethod
    def compute_area(
        coordinates: tuple[Point, Point, Point],
        *,
        signed: bool = False,
    ) -> float:
        if len(coordinates) != 3:
            raise ValueError("Area can be computed only for exactly 3 coordinates.")
        a, b, c = coordinates
        area = (
            a.x * (b.y - c.y) +
            b.x * (c.y - a.y) +
            c.x * (a.y - b.y)
        ) / 2.0
        return area if signed else abs(area)

    def _validate_geometry(self) -> None:
        coordinates = self.coordinates
        if len(coordinates) != 3:
            raise ValueError("Triangle must contain exactly 3 coordinates.")
        for point in coordinates:
            if not isinstance(point, Point):
                raise ValueError("Triangle coordinates must be Point instances.")
            if not _is_finite_number(point.x) or not _is_finite_number(point.y):
                raise ValueError("Triangle coordinates must be finite numeric values.")
        if len(set(coordinates)) != 3:
            raise ValueError("Triangle coordinates must be unique.")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(float(value))
