from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from PySide6.QtGui import QColor, QImage

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.mesh import Mesh
from src.domain.geometry.geometry_utils import mesh_average_quality
from src.infrastructure.image.photo_preprocessor import simplify_contour
from src.infrastructure.processing.image_boundary_extractor import contours_to_boundaries, extract_contours
from src.presentation.tools import TriangulationMode


Mask = list[list[int]]
Contour = list[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class TriangulationSettings:
    target_edge_length: float
    smoothing_iterations: int
    contour_epsilon: float
    max_iterations_factor: int
    min_triangle_quality: float


@dataclass(slots=True)
class TriangulationAdapter:
    MIN_TARGET_EDGE_LENGTH: ClassVar[float] = 4.0
    MAX_TARGET_EDGE_LENGTH: ClassVar[float] = 80.0
    MIN_SMOOTHING_ITERATIONS: ClassVar[int] = 0
    MAX_SMOOTHING_ITERATIONS: ClassVar[int] = 10
    MIN_CONTOUR_EPSILON: ClassVar[float] = 0.5
    MAX_CONTOUR_EPSILON: ClassVar[float] = 10.0
    MIN_MAX_ITERATIONS_FACTOR: ClassVar[int] = 50
    MAX_MAX_ITERATIONS_FACTOR: ClassVar[int] = 1000
    MIN_TRIANGLE_QUALITY: ClassVar[float] = 0.0
    MAX_TRIANGLE_QUALITY: ClassVar[float] = 1.0

    threshold: int = 127
    _mesher: AdvancingFrontMesher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            max_iterations_factor=180,
            target_edge_length=24.0,
            smoothing_iterations=2,
        )

    def run(
        self,
        image_data: QImage,
        mode: TriangulationMode = TriangulationMode.FAST,
        custom_settings: TriangulationSettings | None = None,
    ) -> tuple[Mesh, float]:
        mask = self._qimage_to_mask(image_data)
        contours = extract_contours(mask)
        if not contours:
            raise ValueError("No drawable obstacle found for triangulation.")

        settings = self._resolve_settings(mode, custom_settings)
        selected_contour = max(contours, key=self._contour_area)
        simplified_contour = simplify_contour(selected_contour, epsilon=settings.contour_epsilon)
        if len(simplified_contour) < 3:
            simplified_contour = selected_contour

        boundary = contours_to_boundaries([simplified_contour])[0]
        self._mesher = AdvancingFrontMesher(
            min_triangle_quality=settings.min_triangle_quality,
            max_iterations_factor=settings.max_iterations_factor,
            target_edge_length=settings.target_edge_length,
            smoothing_iterations=settings.smoothing_iterations,
        )
        mesh = self._mesher.generate(boundary)
        coefficient = mesh_average_quality(mesh)
        return mesh, coefficient

    def preset_settings(self, mode: TriangulationMode) -> TriangulationSettings:
        return self._preset_settings(mode)

    def validate_custom_settings(self, settings: TriangulationSettings) -> None:
        if not self.MIN_TARGET_EDGE_LENGTH <= settings.target_edge_length <= self.MAX_TARGET_EDGE_LENGTH:
            raise ValueError(
                f"Target edge length must be in [{self.MIN_TARGET_EDGE_LENGTH}, {self.MAX_TARGET_EDGE_LENGTH}]."
            )
        if not self.MIN_SMOOTHING_ITERATIONS <= settings.smoothing_iterations <= self.MAX_SMOOTHING_ITERATIONS:
            raise ValueError(
                f"Smoothing iterations must be in [{self.MIN_SMOOTHING_ITERATIONS}, {self.MAX_SMOOTHING_ITERATIONS}]."
            )
        if not self.MIN_CONTOUR_EPSILON <= settings.contour_epsilon <= self.MAX_CONTOUR_EPSILON:
            raise ValueError(
                f"Contour simplify epsilon must be in [{self.MIN_CONTOUR_EPSILON}, {self.MAX_CONTOUR_EPSILON}]."
            )
        if not self.MIN_MAX_ITERATIONS_FACTOR <= settings.max_iterations_factor <= self.MAX_MAX_ITERATIONS_FACTOR:
            raise ValueError(
                "Max iterations factor must be in "
                f"[{self.MIN_MAX_ITERATIONS_FACTOR}, {self.MAX_MAX_ITERATIONS_FACTOR}]."
            )
        if not self.MIN_TRIANGLE_QUALITY <= settings.min_triangle_quality <= self.MAX_TRIANGLE_QUALITY:
            raise ValueError(
                f"Min triangle quality must be in [{self.MIN_TRIANGLE_QUALITY}, {self.MAX_TRIANGLE_QUALITY}]."
            )

    def _qimage_to_mask(self, image: QImage) -> Mask:
        grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
        width = grayscale.width()
        height = grayscale.height()
        mask: Mask = []

        for y in range(height):
            row: list[int] = []
            for x in range(width):
                value = QColor(grayscale.pixel(x, y)).value()
                # UI mode triangulates the white domain (background),
                # while black pixels are internal obstacles.
                row.append(1 if value > self.threshold else 0)
            mask.append(row)
        return mask

    def _contour_area(self, contour: Contour) -> float:
        area = 0.0
        n = len(contour)
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _resolve_settings(
        self,
        mode: TriangulationMode,
        custom_settings: TriangulationSettings | None,
    ) -> TriangulationSettings:
        if mode == TriangulationMode.CUSTOM:
            if custom_settings is None:
                raise ValueError("Custom mode selected but custom settings are missing.")
            self.validate_custom_settings(custom_settings)
            return custom_settings
        return self._preset_settings(mode)

    def _preset_settings(self, mode: TriangulationMode) -> TriangulationSettings:
        if mode == TriangulationMode.FAST:
            return TriangulationSettings(
                target_edge_length=28.0,
                smoothing_iterations=2,
                contour_epsilon=3.0,
                max_iterations_factor=180,
                min_triangle_quality=0.01,
            )
        if mode == TriangulationMode.BALANCED:
            return TriangulationSettings(
                target_edge_length=20.0,
                smoothing_iterations=5,
                contour_epsilon=2.0,
                max_iterations_factor=260,
                min_triangle_quality=0.01,
            )
        if mode == TriangulationMode.ACCURATE:
            return TriangulationSettings(
                target_edge_length=14.0,
                smoothing_iterations=8,
                contour_epsilon=1.0,
                max_iterations_factor=400,
                min_triangle_quality=0.01,
            )
        return self._preset_settings(TriangulationMode.BALANCED)
