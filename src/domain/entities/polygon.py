from dataclasses import dataclass

from src.domain.entities.point import Point


@dataclass(slots=True)
class Polygon:
    vertices: list[Point]

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("Polygon vertices must not be empty.")
        if len(self.vertices) < 3:
            raise ValueError("Polygon must contain at least 3 vertices.")
