from abc import ABC, abstractmethod

from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point

class IMeshGenerator(ABC):
    @abstractmethod
    def generate(self, boundary: list[Point]) -> Mesh:
        pass