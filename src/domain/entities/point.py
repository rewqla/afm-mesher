from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float
    # hash((x, y)) computed once: AFM puts the same points into sets millions of times.
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.x, self.y)))

    def __hash__(self) -> int:
        return self._hash
