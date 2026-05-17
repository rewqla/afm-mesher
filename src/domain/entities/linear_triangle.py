from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle

_AREA_EPSILON = 1e-12


@dataclass(frozen=True, slots=True, init=False)
class LinearTriangle(Triangle):
    triangle_number: int
    node_numbers: tuple[int, int, int]
    area: float = field(init=False)
    _phi_i_coeffs: tuple[float, float, float] = field(init=False, repr=False)
    _phi_j_coeffs: tuple[float, float, float] = field(init=False, repr=False)
    _phi_k_coeffs: tuple[float, float, float] = field(init=False, repr=False)

    def __init__(
        self,
        triangle_number: int,
        node_coordinates: tuple[Point, Point, Point],
        node_numbers: tuple[int, int, int],
    ) -> None:
        coordinates = tuple(node_coordinates)
        numbers = tuple(node_numbers)

        if len(coordinates) != 3:
            raise ValueError("LinearTriangle must contain exactly 3 node coordinates.")
        if len(numbers) != 3:
            raise ValueError("LinearTriangle must contain exactly 3 node numbers.")
        if not all(isinstance(node, int) and node > 0 for node in numbers):
            raise ValueError("LinearTriangle node numbers must be positive integers.")
        if len(set(numbers)) != 3:
            raise ValueError("LinearTriangle node numbers must be unique inside one triangle.")
        if not isinstance(triangle_number, int) or triangle_number <= 0:
            raise ValueError("LinearTriangle triangle number must be a positive integer.")

        signed_area = Triangle.compute_area(coordinates, signed=True)
        if abs(signed_area) <= _AREA_EPSILON:
            raise ValueError("LinearTriangle is degenerate (zero area).")
        if signed_area < 0.0:
            coordinates = (coordinates[0], coordinates[2], coordinates[1])
            numbers = (numbers[0], numbers[2], numbers[1])

        super().__init__(a=coordinates[0], b=coordinates[1], c=coordinates[2])

        area = super().area()
        if area <= 0.0:
            raise ValueError("LinearTriangle area must be positive.")
        phi_i, phi_j, phi_k = self._compute_basis_coefficients(self.coordinates, area)

        object.__setattr__(self, "triangle_number", triangle_number)
        object.__setattr__(self, "node_numbers", numbers)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "_phi_i_coeffs", phi_i)
        object.__setattr__(self, "_phi_j_coeffs", phi_j)
        object.__setattr__(self, "_phi_k_coeffs", phi_k)

    @property
    def node_coordinates(self) -> tuple[Point, Point, Point]:
        return self.coordinates

    @property
    def triangle_id(self) -> int:
        return self.triangle_number

    @property
    def node_ids(self) -> tuple[int, int, int]:
        return self.node_numbers

    @property
    def area_value(self) -> float:
        return self.area

    def phi_i(self, x: float, y: float) -> float:
        return self._evaluate_linear(self._phi_i_coeffs, x, y)

    def phi_j(self, x: float, y: float) -> float:
        return self._evaluate_linear(self._phi_j_coeffs, x, y)

    def phi_k(self, x: float, y: float) -> float:
        return self._evaluate_linear(self._phi_k_coeffs, x, y)

    def basis_functions(self) -> tuple:
        return (self.phi_i, self.phi_j, self.phi_k)

    @staticmethod
    def _evaluate_linear(coeffs: tuple[float, float, float], x: float, y: float) -> float:
        if not _is_finite_number(x) or not _is_finite_number(y):
            raise ValueError("Basis function input coordinates must be finite numeric values.")
        alpha, beta, gamma = coeffs
        return alpha + beta * float(x) + gamma * float(y)

    @staticmethod
    def _compute_basis_coefficients(
        node_coordinates: tuple[Point, Point, Point],
        area: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        i, j, k = node_coordinates
        two_delta = 2.0 * area

        phi_i = (
            (j.x * k.y - k.x * j.y) / two_delta,
            (j.y - k.y) / two_delta,
            (k.x - j.x) / two_delta,
        )
        phi_j = (
            (k.x * i.y - i.x * k.y) / two_delta,
            (k.y - i.y) / two_delta,
            (i.x - k.x) / two_delta,
        )
        phi_k = (
            (i.x * j.y - j.x * i.y) / two_delta,
            (i.y - j.y) / two_delta,
            (j.x - i.x) / two_delta,
        )
        return phi_i, phi_j, phi_k


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return isfinite(float(value))
