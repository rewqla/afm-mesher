from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from math import hypot
from typing import ClassVar

from PySide6.QtGui import QColor, QImage

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.application.services.boundary_detection_service import BoundaryDetectionService
from src.application.services.obstacle_processor import ObstacleProcessor
from src.application.services.polygon_builder import PolygonBuilder
from src.application.services.region_classifier import RegionClassifier
from src.application.services.region_topology import ClassifiedRegion, RegionPolygon
from src.domain.entities.point import Point
from src.domain.entities.mesh import Mesh
from src.domain.geometry.geometry_utils import mesh_average_quality
from src.infrastructure.image.photo_preprocessor import simplify_contour
from src.infrastructure.processing.image_boundary_extractor import extract_contours
from src.infrastructure.processing.stroke_centerline_extractor import (
    detect_stroke_branch_points,
    extract_stroke_centerlines,
)
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
    meters_per_pixel: float = 1.0


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
    MIN_METERS_PER_PIXEL: ClassVar[float] = 1e-6
    MAX_METERS_PER_PIXEL: ClassVar[float] = 1e6

    threshold: int = 127
    _mesher: AdvancingFrontMesher = field(init=False, repr=False)
    _boundary_detection: BoundaryDetectionService = field(init=False, repr=False)
    _region_classifier: RegionClassifier = field(init=False, repr=False)
    _obstacle_processor: ObstacleProcessor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            max_iterations_factor=180,
            target_edge_length=24.0,
            smoothing_iterations=2,
        )
        self._boundary_detection = BoundaryDetectionService(PolygonBuilder())
        self._region_classifier = RegionClassifier()
        self._obstacle_processor = ObstacleProcessor()

    def run(
        self,
        image_data: QImage,
        mode: TriangulationMode = TriangulationMode.FAST,
        custom_settings: TriangulationSettings | None = None,
    ) -> tuple[Mesh, float]:
        settings = self._resolve_settings(mode, custom_settings)
        raw_contours = self._extract_geometry_contours(image_data)
        return self._run_with_contours(raw_contours, settings)

    def run_from_contours(
        self,
        contours: list[list[tuple[int, int]]],
        mode: TriangulationMode = TriangulationMode.FAST,
        custom_settings: TriangulationSettings | None = None,
    ) -> tuple[Mesh, float]:
        settings = self._resolve_settings(mode, custom_settings)
        return self._run_with_contours(contours, settings)

    def _run_with_contours(
        self,
        raw_contours: list[Contour],
        settings: TriangulationSettings,
    ) -> tuple[Mesh, float]:
        closed_contours, open_contours = self._split_closed_and_open_contours(raw_contours, settings.contour_epsilon)
        polygons = self._build_polygons(closed_contours, settings.contour_epsilon)
        regions = self._region_classifier.classify(polygons)
        if not regions:
            raise ValueError("No closed contour region found for triangulation.")
        valid_region = self._select_valid_region(regions)
        self._build_topology_graph(valid_region)

        boundary = valid_region.shell.points
        holes = self._obstacle_processor.extract_holes(valid_region)
        holes = self._obstacle_processor.normalize_holes(valid_region.shell, holes)
        cuts = self._extract_cuts(open_contours, settings.contour_epsilon)
        self._mesher = AdvancingFrontMesher(
            min_triangle_quality=settings.min_triangle_quality,
            max_iterations_factor=settings.max_iterations_factor,
            target_edge_length=settings.target_edge_length,
            smoothing_iterations=settings.smoothing_iterations,
        )
        mesh = self._mesher.generate_with_holes_and_cuts(
            boundary,
            [hole.points for hole in holes],
            cuts,
        )
        mesh = Mesh(triangles=mesh.triangles, meters_per_pixel=settings.meters_per_pixel)
        if not mesh.triangles:
            raise ValueError("Triangulation produced no valid triangles for the selected region.")
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
        if not self.MIN_METERS_PER_PIXEL <= settings.meters_per_pixel <= self.MAX_METERS_PER_PIXEL:
            raise ValueError(
                f"Meters per pixel must be in [{self.MIN_METERS_PER_PIXEL}, {self.MAX_METERS_PER_PIXEL}]."
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
                # Geometric contours are drawn in black; convert black regions to 1.
                row.append(1 if value <= self.threshold else 0)
            mask.append(row)
        return mask

    def _extract_geometry_contours(self, image_data: QImage) -> list[Contour]:
        mask = self._qimage_to_mask(image_data)
        contours = extract_contours(mask)
        if not contours:
            raise ValueError("No contour geometry found.")
        return contours

    def extract_contours_from_image(self, image_data: QImage) -> list[Contour]:
        return self._extract_geometry_contours(image_data)

    def extract_stroke_contours_from_image(self, image_data: QImage) -> list[Contour]:
        mask = self._qimage_to_mask(image_data)
        contours = extract_stroke_centerlines(mask)
        if not contours:
            raise ValueError("No stroke geometry found.")
        return contours

    def detect_stroke_branch_points_from_image(self, image_data: QImage) -> list[tuple[int, int]]:
        mask = self._qimage_to_mask(image_data)
        return detect_stroke_branch_points(mask)

    def _build_polygons(self, contours: list[Contour], epsilon: float) -> list[RegionPolygon]:
        simplified: list[Contour] = []
        for contour in contours:
            simplified_contour = simplify_contour(contour, epsilon=epsilon)
            if len(simplified_contour) < 3:
                simplified_contour = contour
            if simplified_contour[0] != simplified_contour[-1]:
                simplified_contour = [*simplified_contour, simplified_contour[0]]
            simplified.append(simplified_contour)
        return self._boundary_detection.detect(simplified)

    def _split_closed_and_open_contours(
        self,
        contours: list[Contour],
        epsilon: float,
    ) -> tuple[list[Contour], list[Contour]]:
        closed_contours: list[Contour] = []
        open_contours: list[Contour] = []

        for contour in contours:
            simplified_contour = simplify_contour(contour, epsilon=epsilon)
            if len(simplified_contour) >= 3 and self._is_closed_contour(contour, simplified_contour, epsilon):
                closed = simplified_contour
                if closed[0] != closed[-1]:
                    closed = [*closed, closed[0]]
                closed_contours.append(closed)
            else:
                open_contours.append(simplified_contour if len(simplified_contour) >= 2 else contour)

        return closed_contours, open_contours

    def _is_closed_contour(self, raw_contour: Contour, simplified_contour: Contour, epsilon: float) -> bool:
        if len(simplified_contour) >= 3 and simplified_contour[0] == simplified_contour[-1]:
            return True
        if len(raw_contour) < 3:
            return False

        closure_tolerance = max(1.0, epsilon * 2.0)
        return hypot(
            float(raw_contour[0][0]) - float(raw_contour[-1][0]),
            float(raw_contour[0][1]) - float(raw_contour[-1][1]),
        ) <= closure_tolerance

    def _extract_cuts(self, contours: list[Contour], epsilon: float) -> list[list[Point]]:
        cuts: list[list[Point]] = []
        for contour in contours:
            if len(contour) < 2:
                continue

            simplified = self._simplify_open_contour(contour, epsilon)
            if len(simplified) < 2:
                continue
            cuts.append([Point(float(x), float(y)) for x, y in simplified])
        return cuts

    def _simplify_open_contour(self, contour: Contour, epsilon: float) -> Contour:
        if len(contour) <= 2:
            return contour
        simplified = simplify_contour(contour, epsilon=epsilon)
        if len(simplified) < 2:
            return contour
        if simplified[0] == simplified[-1]:
            return contour
        return simplified

    def _select_valid_region(self, regions: list[ClassifiedRegion]) -> ClassifiedRegion:
        return max(regions, key=lambda region: self._polygon_area(region.shell.points))

    def _polygon_area(self, contour: list[Point]) -> float:
        area = 0.0
        n = len(contour)
        for i in range(n):
            x1 = contour[i].x
            y1 = contour[i].y
            x2 = contour[(i + 1) % n].x
            y2 = contour[(i + 1) % n].y
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _build_topology_graph(self, region: ClassifiedRegion) -> dict[tuple[float, float], set[tuple[float, float]]]:
        graph: dict[tuple[float, float], set[tuple[float, float]]] = {}
        for polygon in [region.shell, *region.holes]:
            points = polygon.points
            for i in range(len(points)):
                a = (points[i].x, points[i].y)
                b = (points[(i + 1) % len(points)].x, points[(i + 1) % len(points)].y)
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
        return graph

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
        settings = self._preset_settings(mode)
        if custom_settings is None:
            return settings
        settings = replace(settings, meters_per_pixel=custom_settings.meters_per_pixel)
        self.validate_custom_settings(settings)
        return settings

    def _preset_settings(self, mode: TriangulationMode) -> TriangulationSettings:
        if mode == TriangulationMode.FAST:
            return TriangulationSettings(
                target_edge_length=28.0,
                smoothing_iterations=2,
                contour_epsilon=3.0,
                max_iterations_factor=180,
                min_triangle_quality=0.01,
                meters_per_pixel=1.0,
            )
        if mode == TriangulationMode.BALANCED:
            return TriangulationSettings(
                target_edge_length=20.0,
                smoothing_iterations=5,
                contour_epsilon=2.0,
                max_iterations_factor=260,
                min_triangle_quality=0.01,
                meters_per_pixel=1.0,
            )
        if mode == TriangulationMode.ACCURATE:
            return TriangulationSettings(
                target_edge_length=14.0,
                smoothing_iterations=8,
                contour_epsilon=1.0,
                max_iterations_factor=400,
                min_triangle_quality=0.01,
                meters_per_pixel=1.0,
            )
        return self._preset_settings(TriangulationMode.BALANCED)
