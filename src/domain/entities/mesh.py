from dataclasses import dataclass

from src.domain.entities.triangle import Triangle


@dataclass(slots=True)
class Mesh:
    triangles: list[Triangle]

    def __post_init__(self) -> None:
        if not self.triangles:
            raise ValueError("Mesh triangles must not be empty.")
