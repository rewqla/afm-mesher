from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from math import hypot, isclose
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
from src.domain.geometry.geometry_utils import mesh_average_quality, point_in_polygon, segments_intersect
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
        # Validate-by-shell mode: keep closed inner obstacles, ignore open cuts.
        holes = self._collect_inner_obstacles(valid_region.shell, polygons)
        holes = self._obstacle_processor.normalize_holes(valid_region.shell, holes)
        cuts = self._extract_cuts(open_contours, settings.contour_epsilon)
        cuts = self._filter_cuts_conflicting_with_holes(cuts, holes)
        cuts = self._merge_collinear_overlapping_cuts(cuts)
        self._mesher = AdvancingFrontMesher(
            min_triangle_quality=settings.min_triangle_quality,
            max_iterations_factor=settings.max_iterations_factor,
            target_edge_length=settings.target_edge_length,
            smoothing_iterations=settings.smoothing_iterations,
        )
        hole_points = [hole.points for hole in holes]
        attempted_cuts = [
            cuts,
            self._prune_overlapping_collinear_cuts(cuts),
            [],
        ]
        mesh: Mesh | None = None
        last_error: ValueError | None = None
        for candidate_cuts in attempted_cuts:
            try:
                mesh = self._mesher.generate_with_holes_and_cuts(boundary, hole_points, candidate_cuts)
                break
            except ValueError as error:
                last_error = error
                continue
        if mesh is None:
            if last_error is not None:
                raise last_error
            raise ValueError("Triangulation failed.")
        mesh = Mesh(triangles=mesh.triangles, meters_per_pixel=settings.meters_per_pixel)
        if not mesh.triangles:
            raise ValueError("Triangulation produced no valid triangles for the selected region.")
        coefficient = mesh_average_quality(mesh)
        return mesh, coefficient

    def _filter_cuts_conflicting_with_holes(
        self,
        cuts: list[list[Point]],
        holes: list[RegionPolygon],
    ) -> list[list[Point]]:
        if not cuts or not holes:
            return cuts
        filtered: list[list[Point]] = []
        for cut in cuts:
            filtered.extend(self._split_cut_by_hole_conflicts(cut, holes))
        return filtered

    def _split_cut_by_hole_conflicts(self, cut: list[Point], holes: list[RegionPolygon]) -> list[list[Point]]:
        if len(cut) < 2:
            return []

        pieces: list[list[Point]] = []
        for a, b in zip(cut, cut[1:]):
            for seg_a, seg_b in self._clip_segment_outside_holes(a, b, holes):
                pieces.append([seg_a, seg_b])
        return pieces

    def _clip_segment_outside_holes(
        self,
        a: Point,
        b: Point,
        holes: list[RegionPolygon],
    ) -> list[tuple[Point, Point]]:
        t_values = [0.0, 1.0]
        for hole in holes:
            hole_points = hole.points
            for c, d in zip(hole_points, [*hole_points[1:], hole_points[0]]):
                t = self._segment_intersection_t(a, b, c, d)
                if t is not None:
                    t_values.append(t)

        t_values = sorted(t_values)
        deduped_t: list[float] = []
        for t in t_values:
            if not deduped_t or not isclose(t, deduped_t[-1], abs_tol=1e-9):
                deduped_t.append(t)

        kept: list[tuple[Point, Point]] = []
        for t0, t1 in zip(deduped_t, deduped_t[1:]):
            if t1 - t0 <= 1e-9:
                continue
            tm = (t0 + t1) / 2.0
            midpoint = self._point_on_segment(a, b, tm)
            if any(point_in_polygon(midpoint, hole.points, include_boundary=True) for hole in holes):
                continue
            # Avoid exact touching of hole boundaries: trim interval ends that come from intersections.
            seg_len = hypot(b.x - a.x, b.y - a.y)
            offset_t = (0.5 / seg_len) if seg_len > 1e-9 else 0.0
            dt = min(max(offset_t, 1e-4), (t1 - t0) * 0.25)
            seg_t0 = t0 + dt if t0 > 0.0 else t0
            seg_t1 = t1 - dt if t1 < 1.0 else t1
            if seg_t1 - seg_t0 <= 1e-9:
                continue
            p0 = self._point_on_segment(a, b, seg_t0)
            p1 = self._point_on_segment(a, b, seg_t1)
            if hypot(p1.x - p0.x, p1.y - p0.y) > 1e-9:
                kept.append((p0, p1))
        return kept

    def _point_on_segment(self, a: Point, b: Point, t: float) -> Point:
        return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)

    def _segment_intersection_t(self, a: Point, b: Point, c: Point, d: Point) -> float | None:
        if not segments_intersect(a, b, c, d, include_endpoints=True):
            return None
        rx = b.x - a.x
        ry = b.y - a.y
        sx = d.x - c.x
        sy = d.y - c.y
        denom = rx * sy - ry * sx
        if abs(denom) <= 1e-12:
            return None
        cx = c.x - a.x
        cy = c.y - a.y
        t = (cx * sy - cy * sx) / denom
        if -1e-9 <= t <= 1.0 + 1e-9:
            return min(1.0, max(0.0, t))
        return None

    def _merge_collinear_overlapping_cuts(self, cuts: list[list[Point]]) -> list[list[Point]]:
        segments: list[tuple[Point, Point]] = []
        for cut in cuts:
            if len(cut) < 2:
                continue
            for a, b in zip(cut, cut[1:]):
                if hypot(b.x - a.x, b.y - a.y) > 1e-9:
                    segments.append((a, b))

        changed = True
        while changed:
            changed = False
            for i in range(len(segments)):
                if changed:
                    break
                for j in range(i + 1, len(segments)):
                    merged = self._merge_if_collinear_overlapping(segments[i], segments[j])
                    if merged is None:
                        continue
                    new_segments: list[tuple[Point, Point]] = []
                    for k, seg in enumerate(segments):
                        if k in (i, j):
                            continue
                        new_segments.append(seg)
                    new_segments.append(merged)
                    segments = new_segments
                    changed = True
                    break

        return [[a, b] for a, b in segments]

    def _prune_overlapping_collinear_cuts(self, cuts: list[list[Point]]) -> list[list[Point]]:
        segments: list[tuple[Point, Point]] = []
        for cut in cuts:
            if len(cut) < 2:
                continue
            for a, b in zip(cut, cut[1:]):
                if hypot(b.x - a.x, b.y - a.y) > 1e-9:
                    segments.append((a, b))

        segments.sort(key=lambda s: hypot(s[1].x - s[0].x, s[1].y - s[0].y), reverse=True)
        kept: list[tuple[Point, Point]] = []
        for candidate in segments:
            if any(self._merge_if_collinear_overlapping(candidate, existing) is not None for existing in kept):
                continue
            kept.append(candidate)
        return [[a, b] for a, b in kept]

    def _merge_if_collinear_overlapping(
        self,
        first: tuple[Point, Point],
        second: tuple[Point, Point],
    ) -> tuple[Point, Point] | None:
        a, b = first
        c, d = second
        vx = b.x - a.x
        vy = b.y - a.y
        len_sq = vx * vx + vy * vy
        if len_sq <= 1e-12:
            return None

        # Collinearity with pixel-space tolerance (near-collinear strokes should be merged).
        line_len = hypot(vx, vy)
        if line_len <= 1e-9:
            return None
        cross_c = vx * (c.y - a.y) - vy * (c.x - a.x)
        cross_d = vx * (d.y - a.y) - vy * (d.x - a.x)
        dist_c = abs(cross_c) / line_len
        dist_d = abs(cross_d) / line_len
        if dist_c > 0.75 or dist_d > 0.75:
            return None

        def proj_t(p: Point) -> float:
            return ((p.x - a.x) * vx + (p.y - a.y) * vy) / len_sq

        t0, t1 = sorted((0.0, 1.0))
        t2, t3 = sorted((proj_t(c), proj_t(d)))
        overlap_start = max(t0, t2)
        overlap_end = min(t1, t3)
        if overlap_end < overlap_start - 1e-9:
            return None

        merged_start_t = min(t0, t2)
        merged_end_t = max(t1, t3)
        start = self._point_on_segment(a, b, merged_start_t)
        end = self._point_on_segment(a, b, merged_end_t)
        if hypot(end.x - start.x, end.y - start.y) <= 1e-9:
            return None
        return start, end

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

    def _collect_inner_obstacles(self, shell: RegionPolygon, polygons: list[RegionPolygon]) -> list[RegionPolygon]:
        obstacles: list[RegionPolygon] = []
        for polygon in polygons:
            if polygon is shell:
                continue
            if not polygon.points:
                continue
            probe = self._representative_point(polygon.points)
            if point_in_polygon(probe, shell.points, include_boundary=True):
                obstacles.append(polygon)
        return obstacles

    def _representative_point(self, polygon: list[Point]) -> Point:
        if not polygon:
            return Point(0.0, 0.0)

        area2 = 0.0
        cx_acc = 0.0
        cy_acc = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            cross = p1.x * p2.y - p2.x * p1.y
            area2 += cross
            cx_acc += (p1.x + p2.x) * cross
            cy_acc += (p1.y + p2.y) * cross

        if abs(area2) <= 1e-12:
            avg_x = sum(point.x for point in polygon) / len(polygon)
            avg_y = sum(point.y for point in polygon) / len(polygon)
            return Point(avg_x, avg_y)

        factor = 1.0 / (3.0 * area2)
        return Point(cx_acc * factor, cy_acc * factor)

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
