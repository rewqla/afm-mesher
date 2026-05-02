from dataclasses import dataclass

from src.domain.entities.point import Point


@dataclass(frozen=True, slots=True)
class Edge:
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("Edge endpoints must be different points.")
