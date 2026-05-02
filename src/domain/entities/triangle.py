from dataclasses import dataclass
from math import isclose

from src.domain.entities.point import Point

_AREA_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Triangle:
    a: Point
    b: Point
    c: Point

    def __post_init__(self) -> None:
        if self._is_degenerate():
            raise ValueError("Triangle is degenerate (zero area).")

    def area(self) -> float:
        return abs(
            (self.a.x * (self.b.y - self.c.y) +
             self.b.x * (self.c.y - self.a.y) +
             self.c.x * (self.a.y - self.b.y)) / 2.0
        )

    def _is_degenerate(self) -> bool:
        return isclose(self.area(), 0.0, abs_tol=_AREA_EPSILON)
