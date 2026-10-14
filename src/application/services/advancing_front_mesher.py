from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import datetime
from math import ceil, floor, sqrt
from pathlib import Path
from time import monotonic, perf_counter

from src.application.services.mesh_postprocessing import (
    improve_mesh_by_edge_flips,
    laplacian_smooth as postprocess_laplacian_smooth,
    renumber_nodes_rcm_multistart,
    renumber_nodes_rcm,
    renumber_nodes_sloan,
    renumber_nodes_sloan_multistart,
    split_cut_nodes,
)
from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import (
    distance,
    orientation,
    point_in_polygon,
    segments_intersect,
    triangle_area,
    triangle_quality,
)
from src.domain.interfaces.imesh_generator import IMeshGenerator

FrontEdge = tuple[Point, Point]
TriangleIds = tuple[int, int, int]
Segment = tuple[Point, Point]

_EPSILON = 1e-9
_KEY_PRECISION = 10
_MAX_EAR_DIAGONAL_FACTOR = 2.4
_MAX_FRONT_CANDIDATE_SCAN = 256
_MAX_DOMAIN_FRONT_POINTS = 240
_AFM_DEBUG_LOG_PATH = Path.cwd() / "triangulation_debug.log"
DEFAULT_TIME_BUDGET_S = 120.0
# Relative tolerance for "sum of triangle areas == domain area".
_AREA_RELATIVE_TOLERANCE = 1e-6
# Distance (px) under which a point counts as lying on a cut segment / triangle edge.
_CUT_TOUCH_TOLERANCE = 1e-6
# Vertex-on-edge tolerance (pixels, scaled by edge length) for conformity checks.
_ON_EDGE_RELATIVE_TOLERANCE = 1e-7
# Runaway-front guard. A healthy front may grow several times beyond its initial size on fine
# meshes, but it cannot have more edges than a full mesh of the domain with step h (~3*area/h^2).
_MAX_FRONT_GROWTH_FACTOR = 4.0
_MAX_FRONT_GROWTH_MARGIN = 64
_MAX_FRONT_EDGES_PER_CELL = 2.0
# A Steiner point closer than this (times the target step) to a front vertex is replaced by that vertex.
_STEINER_SNAP_FACTOR = 0.5
# Topology cleanup: rounds of (edge flips, valence-3 node removal) and the minimum quality of the
# triangle that replaces three triangles around an interior valence-3 node.
_TOPOLOGY_ROUNDS = 4
_VALENCE3_MERGE_MIN_QUALITY = 0.3


class MeshingTimeoutError(ValueError):
    """Triangulation exceeded its time budget (all passes and cut variants together)."""


class _TriangleSpatialIndex:
    """Uniform grid over triangle bounding boxes for local overlap queries."""

    def __init__(self, cell_size: float) -> None:
        self._cell_size = max(cell_size, 1e-6)
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.triangles: list[tuple[Point, Point, Point]] = []

    def _cell_range(self, points: tuple[Point, ...]) -> tuple[int, int, int, int]:
        c = self._cell_size
        return (
            floor(min(p.x for p in points) / c),
            floor(max(p.x for p in points) / c),
            floor(min(p.y for p in points) / c),
            floor(max(p.y for p in points) / c),
        )

    def insert(self, a: Point, b: Point, c: Point) -> int:
        idx = len(self.triangles)
        self.triangles.append((a, b, c))
        x0, x1, y0, y1 = self._cell_range((a, b, c))
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                self._cells[(gx, gy)].append(idx)
        return idx

    def query(self, a: Point, b: Point, c: Point) -> set[int]:
        x0, x1, y0, y1 = self._cell_range((a, b, c))
        found: set[int] = set()
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                found.update(self._cells.get((gx, gy), ()))
        return found


def _afm_debug(event: str, **fields: object) -> None:
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    payload = " ".join(f"{key}={fields[key]!r}" for key in sorted(fields))
    line = f"{timestamp} [{event}] {payload}\n" if payload else f"{timestamp} [{event}]\n"
    try:
        _AFM_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AFM_DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(line)
    except OSError:
        return


class AdvancingFrontMesher(IMeshGenerator):
    def __init__(
            self,
            min_triangle_quality: float = 0.02,
            max_iterations_factor: int = 500,
            target_edge_length: float | None = None,
            smoothing_iterations: int = 5,
            numbering_strategy: str = "rcm_multistart",
            time_budget_s: float | None = DEFAULT_TIME_BUDGET_S,
    ) -> None:
        if not 0.0 <= min_triangle_quality <= 1.0:
            raise ValueError("min_triangle_quality must be in [0, 1]")
        if max_iterations_factor < 1:
            raise ValueError("max_iterations_factor must be >= 1")
        if target_edge_length is not None and target_edge_length <= 0:
            raise ValueError("target_edge_length must be > 0 when provided")
        if not 0 <= smoothing_iterations <= 20:
            raise ValueError("smoothing_iterations must be in [0, 20]")
        if time_budget_s is not None and time_budget_s <= 0:
            raise ValueError("time_budget_s must be > 0 when provided")

        self._min_triangle_quality = min_triangle_quality
        self._max_iterations_factor = max_iterations_factor
        self._target_edge_length = target_edge_length
        self._smoothing_iterations = smoothing_iterations
        self._numbering_strategy = numbering_strategy
        self._steiner_activation_length_factor = 0.95
        self._debug_progress_interval = 250
        self._time_budget_s = time_budget_s
        self._deadline: float | None = None
        # Triangles already built in the current AFM pass; None outside of a pass.
        self._built_index: _TriangleSpatialIndex | None = None
        # Seconds per stage of the last generate_with_holes_and_cuts call (all passes, failed ones too).
        self.last_stage_timings: dict[str, float] = {"afm": 0.0, "postprocess": 0.0, "numbering": 0.0}

    def generate(self, boundary: list[Point]) -> Mesh:
        return self.generate_with_holes(boundary, holes=[])

    def generate_with_holes(
            self,
            boundary: list[Point],
            holes: list[list[Point]],
            cuts: list[list[Point]] | None = None,
    ) -> Mesh:
        return self.generate_with_holes_and_cuts(boundary, holes, cuts=cuts or [])

    def generate_with_holes_and_cuts(
            self,
            boundary: list[Point],
            holes: list[list[Point]],
            cuts: list[list[Point]],
            deadline: float | None = None,
            merge_inclusions: list[list[Point]] | None = None,
    ) -> Mesh:
        """`deadline` (time.monotonic()) lets a caller share one time budget across several calls;
        without it the mesher's own `time_budget_s` applies to this call.

        `merge_inclusions` are meshed together with the domain: each ring's edges enter the front in
        both directions (like a cut), so both sides share the discretized boundary; its nodes are
        split into outside/inside twins and every triangle gets a subdomain number (0 outside, k for
        the k-th inclusion)."""
        if deadline is None and self._time_budget_s is not None:
            deadline = monotonic() + self._time_budget_s
        self._deadline = deadline
        timings = self.last_stage_timings = {"afm": 0.0, "postprocess": 0.0, "numbering": 0.0}
        polygon = self._prepare_boundary(boundary)
        hole_polygons = [self._prepare_hole_boundary(hole) for hole in holes if hole]
        merge_polygons = [self._prepare_boundary(ring) for ring in (merge_inclusions or []) if ring]
        cut_segments = self._prepare_cuts(cuts, polygon, hole_polygons, tolerance=max(_EPSILON * 100.0, 1e-6))
        # A cut endpoint lying inside a boundary edge must become a boundary vertex; otherwise the
        # boundary front edge passes through the cut (T-junction) and the front cannot close.
        cut_points = [p for segment in cut_segments for p in segment]
        polygon = self._insert_points_on_edges(polygon, cut_points)
        hole_polygons = [self._insert_points_on_edges(hole, cut_points) for hole in hole_polygons]
        base_h = self._resolve_target_step(polygon)
        interface_segments = [edge for ring in merge_polygons for edge in self._polygon_edges(ring)]

        best_mesh: Mesh | None = None
        best_quality = -1.0
        quality_target = 0.7

        attempt_scales = (1.0, 1.2, 0.85, 0.7)
        last_error: ValueError | None = None
        for target_scale in attempt_scales:
            target_h = base_h * target_scale
            stage_start = perf_counter()
            try:
                mesh, outer_boundary_points, interface_points, pass_cut_segments = self._generate_single_pass(
                    polygon,
                    hole_polygons,
                    [*cut_segments, *interface_segments],
                    target_h,
                )
            except MeshingTimeoutError:
                raise
            except ValueError as error:
                last_error = error
                continue
            finally:
                timings["afm"] += perf_counter() - stage_start

            # Postprocessing time excludes the renumbering done inside smooth(), which is timed separately.
            stage_start = perf_counter()
            numbering_before = timings["numbering"]
            mesh = self._improve_topology(
                mesh,
                fixed_points=[*outer_boundary_points, *interface_points],
                cut_segments=pass_cut_segments,
            )
            mesh = self.smooth(
                mesh,
                boundary=outer_boundary_points,
                interface_points=interface_points,
                iterations=self._smoothing_iterations,
                cut_segments=pass_cut_segments,
            )
            self._validate_cut_segments_are_mesh_edges(mesh, pass_cut_segments)
            try:
                self._validate_final_mesh(mesh, self._domain_area(polygon, hole_polygons))
            except ValueError as error:
                _afm_debug("afm.final_validation.failed", target_h=target_h, error=str(error))
                last_error = error
                continue
            finally:
                timings["postprocess"] += perf_counter() - stage_start - (timings["numbering"] - numbering_before)
            avg_quality = self._average_mesh_quality(mesh)

            # Same per-segment subdivision as in the pass, so ring pieces are recognised by key.
            pass_interface = self._subdivide_cut_segments(interface_segments, target_h)
            interface_keys = {self._undirected_edge_key(edge) for edge in pass_interface}
            mesh.cut_edges = tuple(e for e in pass_cut_segments if self._undirected_edge_key(e) not in interface_keys)
            mesh.interface_edges = tuple(pass_interface)
            if merge_polygons and mesh.indexed_mesh_data is not None:
                mesh.indexed_mesh_data.triangle_subdomains = self._triangle_subdomains(
                    mesh.indexed_mesh_data, merge_polygons
                )
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_mesh = mesh
            if avg_quality >= quality_target:
                return mesh

        if best_mesh is None and not hole_polygons and not cut_segments:
            # Last resort for simple domain: robust ear clipping on outer boundary.
            stage_start = perf_counter()
            fallback = self._ear_clip_polygon(polygon, polygon, holes=[], cut_segments=[])
            timings["afm"] += perf_counter() - stage_start
            if fallback:
                fallback_mesh = Mesh(triangles=fallback)
                self._validate_final_mesh(fallback_mesh, self._domain_area(polygon, []))
                return self._apply_rcm_numbering(fallback_mesh, boundary=polygon, interface_points=[])

        if best_mesh is None:
            if last_error is not None:
                raise last_error
            raise ValueError("AFM failed to generate mesh.")
        return best_mesh

    def _generate_single_pass(
            self,
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            target_h: float,
    ) -> tuple[Mesh, list[Point], list[Point], list[Segment]]:
        polygon = self._subdivide_boundary(polygon, target_h)
        holes = [self._subdivide_boundary(hole, target_h) for hole in holes]
        pass_cut_segments = self._subdivide_cut_segments(cut_segments, target_h)
        front = self._build_initial_front(polygon)
        for hole in holes:
            front.extend(self._build_initial_front(hole))
        for a, b in pass_cut_segments:
            front.append((a, b))
            front.append((b, a))
        triangles: list[Triangle] = []
        domain_area = self._domain_area(polygon, holes)
        area_limit = domain_area * (1.0 + _AREA_RELATIVE_TOLERANCE) + _EPSILON
        meshed_area = 0.0
        front_limit = max(
            int(len(front) * _MAX_FRONT_GROWTH_FACTOR) + _MAX_FRONT_GROWTH_MARGIN,
            int(_MAX_FRONT_EDGES_PER_CELL * domain_area / (target_h * target_h)),
        )
        self._built_index = _TriangleSpatialIndex(cell_size=target_h * 1.5)

        max_iterations = max(200, len(front) * self._max_iterations_factor)
        iterations = 0
        _afm_debug(
            "afm.single_pass.start",
            boundary_points=len(polygon),
            holes=len(holes),
            cut_segments=len(pass_cut_segments),
            front_edges=len(front),
            target_h=target_h,
            max_iterations=max_iterations,
        )

        try:
            while front and iterations < max_iterations:
                self._check_deadline()
                if iterations == 0 or iterations % self._debug_progress_interval == 0:
                    _afm_debug(
                        "afm.single_pass.progress",
                        iteration=iterations,
                        front_edges=len(front),
                        triangles=len(triangles),
                    )
                advancement = self._find_advancement(front, polygon, holes, pass_cut_segments, target_h)
                if advancement is None:
                    self._log_stall_diagnostics(front, polygon, holes, pass_cut_segments, target_h)
                    fallback_triangles = self._fallback_triangulate_front(front, polygon, holes, pass_cut_segments)
                    if not fallback_triangles:
                        raise ValueError("AFM stalled: active front cannot be advanced further.")
                    triangles.extend(fallback_triangles)
                    meshed_area += sum(t.area() for t in fallback_triangles)
                    _afm_debug(
                        "afm.single_pass.fallback_success",
                        fallback_triangles=len(fallback_triangles),
                        total_triangles=len(triangles),
                    )
                    front.clear()
                    break

                active_idx, a, b, candidate = advancement
                front.pop(active_idx)

                triangle = Triangle(a, b, candidate)
                triangles.append(triangle)
                self._built_index.insert(a, b, candidate)
                meshed_area += triangle.area()
                self._update_front(front, (a, candidate))
                self._update_front(front, (candidate, b))
                iterations += 1
                if meshed_area > area_limit:
                    _afm_debug("afm.single_pass.area_overflow", meshed_area=meshed_area, domain_area=domain_area)
                    raise ValueError("AFM overlapped already meshed area: triangle area exceeds domain area.")
                if len(front) > front_limit:
                    _afm_debug("afm.single_pass.front_overflow", front_edges=len(front), front_limit=front_limit)
                    raise ValueError("AFM front grew beyond the safe limit (runaway front).")
        finally:
            self._built_index = None

        if front:
            _afm_debug(
                "afm.single_pass.iteration_limit",
                iterations=iterations,
                max_iterations=max_iterations,
                front_edges=len(front),
                triangles=len(triangles),
            )
            raise ValueError("AFM failed to close the front before iteration limit.")
        if not triangles:
            raise ValueError("AFM failed to generate mesh.")

        # A hole (ignore inclusion) is meshed from one side only, so its ring is domain boundary.
        outer_boundary_points = [*polygon]
        for hole in holes:
            outer_boundary_points.extend(hole)
        interface_points: list[Point] = []
        for a, b in pass_cut_segments:
            # Cuts and merge-inclusion rings are glued from both sides: interface nodes.
            interface_points.append(a)
            interface_points.append(b)
        _afm_debug("afm.single_pass.done", triangles=len(triangles))
        return Mesh(triangles=triangles), outer_boundary_points, interface_points, pass_cut_segments

    def _log_stall_diagnostics(
            self,
            front: list[FrontEdge],
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            target_step: float,
    ) -> None:
        max_scan = min(len(front), max(_MAX_FRONT_CANDIDATE_SCAN, len(front) // 3)) if (holes or cut_segments) else min(
            len(front), _MAX_FRONT_CANDIDATE_SCAN
        )
        sorted_indices = self._sorted_front_indices_by_priority(front, max_scan=max_scan)
        all_front_vertices = self._front_vertices(front, exclude=set())
        front_spatial_index, front_cell_size = self._build_segment_spatial_index(front, target_step)
        inspected_edges = min(len(sorted_indices), 24)

        reason_counts: dict[str, int] = defaultdict(int)
        total_candidates = 0
        candidate_breakdown = {"steiner": 0, "near": 0, "far": 0}

        for idx in sorted_indices[:inspected_edges]:
            a, b = front[idx]
            edge_len = distance(a, b)
            if edge_len <= _EPSILON:
                reason_counts["degenerate_base_edge"] += 1
                continue

            midpoint = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
            existing_vertices = [p for p in all_front_vertices if p != a and p != b]
            steiner_candidates: list[Point] = []
            if edge_len > target_step * self._steiner_activation_length_factor:
                steiner_candidates = self._steiner_candidates(a, b)
            near_vertices = [p for p in existing_vertices if distance(p, midpoint) <= edge_len + _EPSILON]
            near_vertices.sort(key=lambda p: distance(p, midpoint))
            far_vertices = [p for p in existing_vertices if distance(p, midpoint) <= edge_len * 2.5]
            far_vertices.sort(key=lambda p: distance(p, midpoint))

            for p in steiner_candidates[:18]:
                candidate_breakdown["steiner"] += 1
                total_candidates += 1
                if self._is_too_close_to_existing_edges(
                        p,
                        front,
                        threshold=0.4 * target_step,
                        spatial_index=front_spatial_index,
                        cell_size=front_cell_size,
                ):
                    reason_counts["too_close_to_front_edge"] += 1
                    continue
                reason = self._triangle_invalid_reason(
                    a, b, p, polygon, holes, cut_segments, front, front_spatial_index, front_cell_size
                )
                if reason is None:
                    reason_counts["valid_found_but_not_selected"] += 1
                else:
                    reason_counts[reason] += 1

            for p in near_vertices[:24]:
                candidate_breakdown["near"] += 1
                total_candidates += 1
                if not self._is_reasonable_closure_candidate(a, b, p, edge_len, max_ratio=1.8):
                    reason_counts["unreasonable_near_closure"] += 1
                    continue
                reason = self._triangle_invalid_reason(
                    a, b, p, polygon, holes, cut_segments, front, front_spatial_index, front_cell_size
                )
                reason_counts["valid_found_but_not_selected" if reason is None else reason] += 1

            for p in far_vertices[:24]:
                candidate_breakdown["far"] += 1
                total_candidates += 1
                if not self._is_reasonable_closure_candidate(a, b, p, edge_len, max_ratio=2.2):
                    reason_counts["unreasonable_far_closure"] += 1
                    continue
                reason = self._triangle_invalid_reason(
                    a, b, p, polygon, holes, cut_segments, front, front_spatial_index, front_cell_size
                )
                reason_counts["valid_found_but_not_selected" if reason is None else reason] += 1

        _afm_debug(
            "afm.stall_diagnostics",
            front_edges=len(front),
            holes=len(holes),
            cut_segments=len(cut_segments),
            target_step=target_step,
            scanned_edges=inspected_edges,
            total_candidates=total_candidates,
            steiner_candidates=candidate_breakdown["steiner"],
            near_candidates=candidate_breakdown["near"],
            far_candidates=candidate_breakdown["far"],
            rejection_reasons=dict(sorted(reason_counts.items())),
        )

    def _triangle_invalid_reason(
            self,
            a: Point,
            b: Point,
            c: Point,
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            front: list[FrontEdge],
            front_spatial_index: dict[tuple[int, int], list[int]] | None = None,
            front_cell_size: float | None = None,
    ) -> str | None:
        if c == a or c == b:
            return "same_vertex"
        if orientation(a, b, c) <= 0:
            return "non_ccw"
        if not self._point_in_domain(c, polygon, holes):
            return "candidate_outside_domain"

        centroid = Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
        if not self._point_in_domain(centroid, polygon, holes):
            return "centroid_outside_domain"
        if not self._triangle_respects_holes(a, b, c, holes):
            return "violates_holes"
        if not self._triangle_respects_cut_segments(a, b, c, cut_segments):
            return "violates_cuts"
        try:
            quality = triangle_quality(a, b, c)
        except ValueError:
            return "degenerate_triangle"
        if quality < self._min_triangle_quality:
            return "quality_below_threshold"

        for new_edge in ((a, c), (c, b)):
            if self._edge_intersects_front(
                    new_edge,
                    front,
                    spatial_index=front_spatial_index,
                    cell_size=front_cell_size,
            ):
                return "intersects_front"

        if self._contains_front_vertex(a, b, c, front):
            return "contains_front_vertex"
        if self._overlaps_built_triangles(a, b, c):
            return "overlaps_built_triangles"
        return None

    def _prepare_boundary(self, boundary: list[Point]) -> list[Point]:
        if len(boundary) < 3:
            raise ValueError("Boundary must contain at least 3 points.")

        cleaned: list[Point] = []
        for point in boundary:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(point)

        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        if len(cleaned) < 3:
            raise ValueError("Boundary must contain at least 3 distinct points.")

        area = self._signed_area(cleaned)
        if abs(area) <= _EPSILON:
            raise ValueError("Boundary polygon has zero area.")

        # CCW boundary so interior is on the left side of front edges.
        if area < 0:
            cleaned.reverse()
        return cleaned

    def _prepare_hole_boundary(self, boundary: list[Point]) -> list[Point]:
        cleaned = self._prepare_boundary(boundary)
        if self._signed_area(cleaned) > 0:
            cleaned.reverse()
        return cleaned

    def _prepare_cuts(
            self,
            cuts: list[list[Point]],
            boundary: list[Point],
            holes: list[list[Point]],
            tolerance: float,
    ) -> list[Segment]:
        domain_vertices = [*boundary]
        for hole in holes:
            domain_vertices.extend(hole)

        normalized: list[list[Point]] = []
        for cut in cuts:
            cleaned = self._normalize_cut_polyline(cut, tolerance)
            if len(cleaned) < 2:
                continue
            snapped = [self._snap_point_to_vertices(p, domain_vertices, tolerance) for p in cleaned]
            self._validate_cut_polyline(snapped, boundary, holes, tolerance)
            normalized.append(snapped)

        segments: list[Segment] = []
        for cut in normalized:
            for i in range(len(cut) - 1):
                a = cut[i]
                b = cut[i + 1]
                if distance(a, b) <= tolerance:
                    continue
                segments.append((a, b))

        return self._split_cut_segments(segments, tolerance)

    def _resolve_target_step(self, boundary: list[Point]) -> float:
        if self._target_edge_length is not None:
            perimeter = 0.0
            for i in range(len(boundary)):
                perimeter += distance(boundary[i], boundary[(i + 1) % len(boundary)])
            min_h_by_front_budget = max(perimeter / _MAX_DOMAIN_FRONT_POINTS, 1e-3)
            return max(self._target_edge_length, min_h_by_front_budget)

        perimeter = 0.0
        for i in range(len(boundary)):
            perimeter += distance(boundary[i], boundary[(i + 1) % len(boundary)])
        h_avg = perimeter / len(boundary)
        return max(h_avg, 1e-3)

    def _cut_vertex_normal(self, cut: list[Point], index: int) -> Point:
        if len(cut) == 2:
            direction = self._unit_direction(cut[0], cut[1])
            return Point(-direction.y, direction.x)

        if index == 0:
            direction = self._unit_direction(cut[0], cut[1])
            return Point(-direction.y, direction.x)
        if index == len(cut) - 1:
            direction = self._unit_direction(cut[-2], cut[-1])
            return Point(-direction.y, direction.x)

        prev_dir = self._unit_direction(cut[index - 1], cut[index])
        next_dir = self._unit_direction(cut[index], cut[index + 1])
        nx = -prev_dir.y - next_dir.y
        ny = prev_dir.x + next_dir.x
        length = sqrt(nx * nx + ny * ny)
        if length <= _EPSILON:
            return Point(-prev_dir.y, prev_dir.x)
        return Point(nx / length, ny / length)

    def _unit_direction(self, a: Point, b: Point) -> Point:
        dx = b.x - a.x
        dy = b.y - a.y
        length = sqrt(dx * dx + dy * dy)
        if length <= _EPSILON:
            return Point(1.0, 0.0)
        return Point(dx / length, dy / length)

    def subdivide_boundary(self, points: list[Point], h: float) -> list[Point]:
        return self._subdivide_boundary(points, h)

    def _subdivide_boundary(self, points: list[Point], h: float) -> list[Point]:
        """
        Uniform boundary subdivision by target step h.
        For each edge with length L, split into k = ceil(L/h) equal segments.
        """
        if h <= 0:
            raise ValueError("h must be > 0")
        if len(points) < 3:
            raise ValueError("Boundary must contain at least 3 points.")

        subdivided: list[Point] = []
        n = len(points)

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            if not subdivided:
                subdivided.append(p1)

            edge_len = distance(p1, p2)
            k = max(1, ceil(edge_len / h))
            for j in range(1, k):
                t = j / k
                subdivided.append(
                    Point(
                        p1.x + (p2.x - p1.x) * t,
                        p1.y + (p2.y - p1.y) * t,
                        )
                )
            subdivided.append(p2)

        if subdivided and distance(subdivided[0], subdivided[-1]) <= _EPSILON:
            subdivided.pop()

        # Remove accidental duplicates.
        deduped: list[Point] = []
        for point in subdivided:
            if not deduped or distance(deduped[-1], point) > _EPSILON:
                deduped.append(point)

        if len(deduped) < 3:
            return points
        return deduped

    def _build_initial_front(self, polygon: list[Point]) -> list[FrontEdge]:
        return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]

    def _sorted_front_indices_by_priority(self, front: list[FrontEdge], max_scan: int | None = None) -> list[int]:
        indices = range(len(front))
        if max_scan is None or max_scan >= len(front):
            return sorted(indices, key=lambda i: distance(front[i][0], front[i][1]))
        return heapq.nsmallest(max_scan, indices, key=lambda i: distance(front[i][0], front[i][1]))

    def _find_advancement(
            self,
            front: list[FrontEdge],
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            target_step: float,
    ) -> tuple[int, Point, Point, Point] | None:
        if holes or cut_segments:
            max_scan = min(len(front), max(_MAX_FRONT_CANDIDATE_SCAN, len(front) // 3))
        else:
            max_scan = min(len(front), _MAX_FRONT_CANDIDATE_SCAN)
        sorted_indices = self._sorted_front_indices_by_priority(front, max_scan=max_scan)
        all_front_vertices = self._front_vertices(front, exclude=set())
        front_spatial_index, front_cell_size = self._build_segment_spatial_index(front, target_step)

        for idx in sorted_indices:
            a, b = front[idx]
            candidate = self._find_best_node(
                a,
                b,
                polygon,
                holes,
                cut_segments,
                front,
                target_step,
                all_front_vertices,
                front_spatial_index,
                front_cell_size,
            )
            if candidate is not None:
                return idx, a, b, candidate
        # Recovery path: if no candidate found for edge direction (a->b),
        # try reversed orientation (b->a). This helps on residual fronts
        # where local edge orientation becomes inconsistent near closure.
        # The reversed triangle passes the same validation (including the overlap
        # check against built triangles), so it cannot re-fill already meshed area.
        for idx in sorted_indices:
            a, b = front[idx]
            candidate = self._find_best_node(
                b,
                a,
                polygon,
                holes,
                cut_segments,
                front,
                target_step,
                all_front_vertices,
                front_spatial_index,
                front_cell_size,
            )
            if candidate is not None:
                _afm_debug("afm.recovery.reversed_edge_used", edge_idx=idx)
                return idx, b, a, candidate
        return None

    def _find_best_node(
            self,
            a: Point,
            b: Point,
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            front: list[FrontEdge],
            target_step: float,
            all_front_vertices: list[Point] | None = None,
            front_spatial_index: dict[tuple[int, int], list[int]] | None = None,
            front_cell_size: float | None = None,
    ) -> Point | None:
        edge_len = distance(a, b)
        if edge_len <= _EPSILON:
            return None

        midpoint = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
        if all_front_vertices is None:
            existing_vertices = self._front_vertices(front, exclude={a, b})
        else:
            existing_vertices = [p for p in all_front_vertices if p != a and p != b]

        # 1) Prefer interior growth via Steiner points.
        if edge_len > target_step * self._steiner_activation_length_factor:
            for p in self._steiner_candidates(a, b):
                snapped = self._nearest_front_vertex(
                    p,
                    radius=_STEINER_SNAP_FACTOR * target_step,
                    exclude=(a, b),
                    front=front,
                    spatial_index=front_spatial_index,
                    cell_size=front_cell_size,
                )
                if snapped is not None:
                    if self._is_reasonable_closure_candidate(a, b, snapped, edge_len, max_ratio=1.8) and self._is_valid_triangle(
                            a,
                            b,
                            snapped,
                            polygon,
                            holes,
                            cut_segments,
                            front,
                            front_spatial_index=front_spatial_index,
                            front_cell_size=front_cell_size,
                    ):
                        return snapped
                    continue
                if self._is_too_close_to_existing_edges(
                        p,
                        front,
                        threshold=0.4 * target_step,
                        spatial_index=front_spatial_index,
                        cell_size=front_cell_size,
                ):
                    continue
                if self._is_valid_triangle(
                        a,
                        b,
                        p,
                        polygon,
                        holes,
                        cut_segments,
                        front,
                        front_spatial_index=front_spatial_index,
                        front_cell_size=front_cell_size,
                ):
                    return p

        # Distances to the edge midpoint are computed once and shared by the near (radius L)
        # and far (radius 2.5L) closure passes; the stable sort keeps the original order.
        closure_radius = edge_len * 2.5
        by_distance: list[tuple[float, Point]] = []
        for p in existing_vertices:
            d = sqrt((midpoint.x - p.x) ** 2 + (midpoint.y - p.y) ** 2)
            if d <= closure_radius:
                by_distance.append((d, p))
        by_distance.sort(key=lambda item: item[0])

        # 2) Try to close topology with existing nodes within local radius L.
        near_vertices = [p for d, p in by_distance if d <= edge_len + _EPSILON]
        for p in near_vertices:
            if not self._is_reasonable_closure_candidate(a, b, p, edge_len, max_ratio=1.8):
                continue
            if self._is_valid_triangle(
                    a,
                    b,
                    p,
                    polygon,
                    holes,
                    cut_segments,
                    front,
                    front_spatial_index=front_spatial_index,
                    front_cell_size=front_cell_size,
            ):
                return p

        # 3) Wider closure radius for final wave connection.
        far_vertices = [p for _, p in by_distance]
        for p in far_vertices:
            if not self._is_reasonable_closure_candidate(a, b, p, edge_len, max_ratio=2.2):
                continue
            if self._is_valid_triangle(
                    a,
                    b,
                    p,
                    polygon,
                    holes,
                    cut_segments,
                    front,
                    front_spatial_index=front_spatial_index,
                    front_cell_size=front_cell_size,
            ):
                return p

        return None

    def _is_too_close_to_existing_edges(
            self,
            point: Point,
            edges: list[FrontEdge],
            threshold: float,
            spatial_index: dict[tuple[int, int], list[int]] | None = None,
            cell_size: float | None = None,
    ) -> bool:
        candidate_ids: list[int] | None = None
        if spatial_index is not None and cell_size is not None and cell_size > _EPSILON:
            probe = threshold + cell_size
            candidate_ids = self._query_segment_spatial_index(
                spatial_index,
                Point(point.x - probe, point.y - probe),
                Point(point.x + probe, point.y + probe),
                cell_size,
            )
        iterable = candidate_ids if candidate_ids is not None else range(len(edges))
        for idx in iterable:
            edge = edges[idx]
            if self._distance_point_to_segment(point, edge[0], edge[1]) < threshold:
                return True
        return False

    def _distance_point_to_segment(self, p: Point, a: Point, b: Point) -> float:
        dx = b.x - a.x
        dy = b.y - a.y
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= _EPSILON:
            return distance(p, a)
        t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj = Point(a.x + t * dx, a.y + t * dy)
        return distance(p, proj)

    def _steiner_candidates(self, a: Point, b: Point) -> list[Point]:
        edge_len = distance(a, b)
        if edge_len <= _EPSILON:
            return []

        mx = (a.x + b.x) / 2.0
        my = (a.y + b.y) / 2.0
        dx = (b.x - a.x) / edge_len
        dy = (b.y - a.y) / edge_len
        nx = -dy
        ny = dx
        height = sqrt(3.0) * edge_len / 2.0

        candidates: list[Point] = []
        # Low apexes (0.5 of the ideal height) gave flat triangles that neighbouring edges closed
        # onto right away, leaving interior valence-3 nodes ("triangles in triangles").
        for h_scale in (1.0, 0.75):
            for t_shift in (0.0, 0.25, -0.25):
                candidates.append(
                    Point(
                        mx + dx * edge_len * t_shift + nx * height * h_scale,
                        my + dy * edge_len * t_shift + ny * height * h_scale,
                        )
                )
        return candidates

    def _is_valid_triangle(
            self,
            a: Point,
            b: Point,
            c: Point,
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
            front: list[FrontEdge],
            front_spatial_index: dict[tuple[int, int], list[int]] | None = None,
            front_cell_size: float | None = None,
    ) -> bool:
        if c == a or c == b:
            return False
        if orientation(a, b, c) <= 0:
            return False
        if not self._point_in_domain(c, polygon, holes):
            return False

        centroid = Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
        if not self._point_in_domain(centroid, polygon, holes):
            return False
        if not self._triangle_respects_holes(a, b, c, holes):
            return False
        if not self._triangle_respects_cut_segments(a, b, c, cut_segments):
            return False

        try:
            quality = triangle_quality(a, b, c)
        except ValueError:
            return False
        if quality < self._min_triangle_quality:
            return False

        for new_edge in ((a, c), (c, b)):
            if self._edge_intersects_front(
                    new_edge,
                    front,
                    spatial_index=front_spatial_index,
                    cell_size=front_cell_size,
            ):
                return False

        if self._contains_front_vertex(a, b, c, front):
            return False
        if self._overlaps_built_triangles(a, b, c):
            return False
        return True

    def _point_in_domain(
            self,
            point: Point,
            polygon: list[Point],
            holes: list[list[Point]],
    ) -> bool:
        if not point_in_polygon(point, polygon, include_boundary=True):
            return False
        if any(point_in_polygon(point, hole, include_boundary=False) for hole in holes):
            return False
        return True

    def _triangle_respects_holes(
            self,
            a: Point,
            b: Point,
            c: Point,
            holes: list[list[Point]],
    ) -> bool:
        triangle_edges = ((a, b), (b, c), (c, a))
        for obstacle in holes:
            if any(point_in_polygon(point, obstacle, include_boundary=False) for point in (a, b, c)):
                return False

            for triangle_edge in triangle_edges:
                for obstacle_edge in self._polygon_edges(obstacle):
                    if self._same_edge(triangle_edge, obstacle_edge):
                        continue
                    if segments_intersect(
                            triangle_edge[0],
                            triangle_edge[1],
                            obstacle_edge[0],
                            obstacle_edge[1],
                            include_endpoints=False,
                    ):
                        return False

            for point in obstacle:
                if point in (a, b, c):
                    continue
                if self._point_in_triangle_strict(point, a, b, c):
                    return False
        return True

    def _triangle_respects_cut_segments(self, a: Point, b: Point, c: Point, cut_segments: list[Segment]) -> bool:
        triangle_edges = ((a, b), (b, c), (c, a))
        for edge in triangle_edges:
            for cut_segment in cut_segments:
                if self._same_undirected_edge(edge, cut_segment):
                    continue
                if segments_intersect(edge[0], edge[1], cut_segment[0], cut_segment[1], include_endpoints=False):
                    return False
                # Touching cases `include_endpoints=False` lets through: an edge running along the cut
                # (collinear overlap) or passing through a cut node, or a triangle vertex in a cut's interior.
                if self._point_strictly_inside_segment(cut_segment[0], edge[0], edge[1]):
                    return False
                if self._point_strictly_inside_segment(cut_segment[1], edge[0], edge[1]):
                    return False
                if self._point_strictly_inside_segment(edge[0], cut_segment[0], cut_segment[1]):
                    return False
        for cut_segment in cut_segments:
            # A cut segment starting at a triangle vertex must not run into the triangle's interior.
            mid = Point((cut_segment[0].x + cut_segment[1].x) / 2.0, (cut_segment[0].y + cut_segment[1].y) / 2.0)
            sides = {orientation(a, b, mid), orientation(b, c, mid), orientation(c, a, mid)}
            if sides == {1} or sides == {-1}:
                return False
        return True

    def _point_strictly_inside_segment(self, p: Point, a: Point, b: Point) -> bool:
        if p == a or p == b:
            return False
        length = distance(a, b)
        if length <= _EPSILON:
            return False
        if abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / length > _CUT_TOUCH_TOLERANCE:
            return False
        t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / (length * length)
        return _CUT_TOUCH_TOLERANCE / length < t < 1.0 - _CUT_TOUCH_TOLERANCE / length

    def _polygon_edges(self, polygon: list[Point]) -> list[FrontEdge]:
        return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]

    def _edge_intersects_front(
            self,
            edge: FrontEdge,
            front: list[FrontEdge],
            spatial_index: dict[tuple[int, int], list[int]] | None = None,
            cell_size: float | None = None,
    ) -> bool:
        candidate_ids: list[int] | None = None
        if spatial_index is not None and cell_size is not None and cell_size > _EPSILON:
            candidate_ids = self._query_segment_spatial_index(spatial_index, edge[0], edge[1], cell_size)
        iterable = candidate_ids if candidate_ids is not None else range(len(front))
        for idx in iterable:
            current = front[idx]
            if not segments_intersect(edge[0], edge[1], current[0], current[1], include_endpoints=False):
                continue
            if self._share_endpoint(edge, current):
                continue
            return True
        return False

    def _build_segment_spatial_index(
            self,
            segments: list[Segment],
            scale_hint: float,
    ) -> tuple[dict[tuple[int, int], list[int]], float]:
        cell_size = max(scale_hint * 1.5, 1e-6)
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (a, b) in enumerate(segments):
            min_x = min(a.x, b.x)
            max_x = max(a.x, b.x)
            min_y = min(a.y, b.y)
            max_y = max(a.y, b.y)
            x0 = int(min_x // cell_size)
            x1 = int(max_x // cell_size)
            y0 = int(min_y // cell_size)
            y1 = int(max_y // cell_size)
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    grid[(gx, gy)].append(idx)
        return grid, cell_size

    def _query_segment_spatial_index(
            self,
            grid: dict[tuple[int, int], list[int]],
            a: Point,
            b: Point,
            cell_size: float,
    ) -> list[int]:
        min_x = min(a.x, b.x)
        max_x = max(a.x, b.x)
        min_y = min(a.y, b.y)
        max_y = max(a.y, b.y)
        x0 = int(min_x // cell_size)
        x1 = int(max_x // cell_size)
        y0 = int(min_y // cell_size)
        y1 = int(max_y // cell_size)
        out: set[int] = set()
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                out.update(grid.get((gx, gy), []))
        return list(out)

    def _contains_front_vertex(self, a: Point, b: Point, c: Point, front: list[FrontEdge]) -> bool:
        """A front vertex strictly inside the triangle or inside one of its sides (hanging node)."""
        for p in self._front_vertices(front, exclude={a, b, c}):
            o1 = orientation(a, b, p)
            o2 = orientation(b, c, p)
            o3 = orientation(c, a, p)
            if o1 > 0 and o2 > 0 and o3 > 0:
                return True
            if o1 >= 0 and o2 >= 0 and o3 >= 0 and (
                    self._point_inside_segment(p, a, b)
                    or self._point_inside_segment(p, b, c)
                    or self._point_inside_segment(p, c, a)
            ):
                return True
        return False

    def _front_vertices(self, front: list[FrontEdge], exclude: set[Point]) -> list[Point]:
        vertices = {edge[0] for edge in front}
        vertices.update(edge[1] for edge in front)
        return [p for p in vertices if p not in exclude]

    def _update_front(self, front: list[FrontEdge], new_edge: FrontEdge) -> None:
        reverse = (new_edge[1], new_edge[0])
        for i, edge in enumerate(front):
            if self._same_edge(edge, reverse):
                front.pop(i)
                return
        front.append(new_edge)

    def _same_edge(self, e1: FrontEdge, e2: FrontEdge) -> bool:
        return distance(e1[0], e2[0]) <= _EPSILON and distance(e1[1], e2[1]) <= _EPSILON

    def _same_undirected_edge(self, e1: FrontEdge, e2: FrontEdge) -> bool:
        return self._same_edge(e1, e2) or self._same_edge(e1, (e2[1], e2[0]))

    def _share_endpoint(self, e1: FrontEdge, e2: FrontEdge) -> bool:
        return e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]

    def smooth(
            self,
            mesh: Mesh,
            boundary: list[Point],
            interface_points: list[Point] | None = None,
            iterations: int = 5,
            cut_segments: list[Segment] | None = None,
    ) -> Mesh:
        return self._laplacian_smooth(mesh, boundary, interface_points or [], iterations, cut_segments or [])

    def _laplacian_smooth(
            self,
            mesh: Mesh,
            boundary: list[Point],
            interface_points: list[Point],
            iterations: int,
            cut_segments: list[Segment] | None = None,
    ) -> Mesh:
        # Smoothing works on the coordinate-glued topology (cut nodes are fixed interface nodes);
        # cut nodes are split into per-side twins only afterwards, in _apply_rcm_numbering.
        node_positions, triangles, _, _, boundary_ids, interface_ids = self._build_topology(
            mesh,
            boundary,
            interface_points,
        )
        if not node_positions:
            return self._apply_rcm_numbering(mesh, cut_segments=cut_segments)

        ordered_node_ids = sorted(node_positions)
        nodes = [(node_positions[node_id].x, node_positions[node_id].y) for node_id in ordered_node_ids]
        smoothed_nodes = postprocess_laplacian_smooth(
            nodes,
            triangles,
            boundary_ids,
            interface_nodes=interface_ids,
            iterations=iterations,
        )
        smoothed_positions = {
            node_id: Point(*smoothed_nodes[node_id - 1])
            for node_id in ordered_node_ids
        }
        smoothed_triangles = [
            Triangle(smoothed_positions[a], smoothed_positions[b], smoothed_positions[c])
            for a, b, c in triangles
        ]
        smoothed_mesh = Mesh(triangles=smoothed_triangles, meters_per_pixel=mesh.meters_per_pixel)
        return self._apply_rcm_numbering(
            smoothed_mesh,
            boundary=boundary,
            interface_points=interface_points,
            cut_segments=cut_segments,
        )

    def _apply_rcm_numbering(
        self,
        mesh: Mesh,
        boundary: list[Point] | None = None,
        interface_points: list[Point] | None = None,
        numbering_strategy: str | None = None,
        cut_segments: list[Segment] | None = None,
    ) -> Mesh:
        linear_triangles = mesh.linear_triangles(key_precision=_KEY_PRECISION)
        node_coordinates_by_id: dict[int, tuple[float, float]] = {}
        indexed_triangles: list[tuple[int, int, int]] = []

        for linear_triangle in linear_triangles:
            indexed_triangles.append(linear_triangle.node_numbers)
            for node_id, point in zip(linear_triangle.node_numbers, linear_triangle.node_coordinates):
                node_coordinates_by_id.setdefault(node_id, (point.x, point.y))

        ordered_nodes = [node_coordinates_by_id[node_id] for node_id in sorted(node_coordinates_by_id)]
        ordered_nodes, indexed_triangles, twin_groups = self._split_cut_nodes(
            ordered_nodes, indexed_triangles, cut_segments or []
        )
        numbering_strategy = self._numbering_strategy if numbering_strategy is None else numbering_strategy
        numbering_start = perf_counter()
        if numbering_strategy == "rcm_multistart":
            rcm_nodes, rcm_triangles, bandwidth = renumber_nodes_rcm_multistart(ordered_nodes, indexed_triangles)
        elif numbering_strategy == "rcm":
            rcm_nodes, rcm_triangles, bandwidth = renumber_nodes_rcm(ordered_nodes, indexed_triangles)
        elif numbering_strategy == "sloan":
            rcm_nodes, rcm_triangles, bandwidth = renumber_nodes_sloan(ordered_nodes, indexed_triangles)
        elif numbering_strategy == "sloan_multistart":
            rcm_nodes, rcm_triangles, bandwidth = renumber_nodes_sloan_multistart(ordered_nodes, indexed_triangles)
        else:
            raise ValueError(
                "numbering_strategy must be one of 'rcm', 'rcm_multistart', 'sloan', or 'sloan_multistart'"
            )
        self.last_stage_timings["numbering"] += perf_counter() - numbering_start
        # Renumbering keeps the triangle order, so the old->new node map is read off the triangles.
        old_to_new = {
            old_id: new_id
            for old_triangle, new_triangle in zip(indexed_triangles, rcm_triangles)
            for old_id, new_id in zip(old_triangle, new_triangle)
        }
        twin_groups = [tuple(sorted(old_to_new[node_id] for node_id in group)) for group in twin_groups]
        boundary_nodes: set[int] = set()
        interface_nodes: set[int] = set()
        # Twins share coordinates: a point resolves to every number it has.
        node_lookup: dict[tuple[float, float], list[int]] = defaultdict(list)
        for idx, (x, y) in enumerate(rcm_nodes, start=1):
            node_lookup[(round(x, _KEY_PRECISION), round(y, _KEY_PRECISION))].append(idx)
        if boundary is not None:
            for point in boundary:
                boundary_nodes.update(
                    node_lookup.get((round(point.x, _KEY_PRECISION), round(point.y, _KEY_PRECISION)), ())
                )
        if interface_points is not None:
            for point in interface_points:
                interface_nodes.update(
                    node_lookup.get((round(point.x, _KEY_PRECISION), round(point.y, _KEY_PRECISION)), ())
                )
        _afm_debug("afm.postprocess.rcm", nodes=len(rcm_nodes), bandwidth=bandwidth, strategy=numbering_strategy)
        indexed_mesh = IndexedMesh(
            nodes=rcm_nodes,
            triangles=rcm_triangles,
            boundary_nodes=boundary_nodes,
            interface_nodes=interface_nodes,
            bandwidth=bandwidth,
            index_base=1,
            meters_per_pixel=mesh.meters_per_pixel,
            twin_groups=twin_groups,
        )
        return Mesh(
            triangles=mesh.triangles,
            meters_per_pixel=mesh.meters_per_pixel,
            node_order=tuple(Point(x, y) for x, y in rcm_nodes),
            indexed_mesh_data=indexed_mesh,
        )

    def _split_cut_nodes(
        self,
        nodes: list[tuple[float, float]],
        triangles: list[tuple[int, int, int]],
        cut_segments: list[Segment],
    ) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], list[tuple[int, ...]]]:
        if not cut_segments:
            return nodes, triangles, []
        node_ids = {
            (round(x, _KEY_PRECISION), round(y, _KEY_PRECISION)): idx
            for idx, (x, y) in enumerate(nodes, start=1)
        }
        cut_edges: set[frozenset[int]] = set()
        for a, b in cut_segments:
            a_id = node_ids.get(self._point_key(a))
            b_id = node_ids.get(self._point_key(b))
            if a_id is not None and b_id is not None:
                cut_edges.add(frozenset((a_id, b_id)))
        return split_cut_nodes(nodes, triangles, cut_edges)

    def _triangle_subdomains(self, indexed_mesh: IndexedMesh, rings: list[list[Point]]) -> list[int]:
        """0 for the main domain, k (1-based) for a triangle inside the k-th merge inclusion."""
        nodes, base = indexed_mesh.nodes, indexed_mesh.index_base
        subdomains: list[int] = []
        for triangle in indexed_mesh.triangles:
            xs, ys = zip(*(nodes[node_id - base] for node_id in triangle))
            centroid = Point(sum(xs) / 3.0, sum(ys) / 3.0)
            subdomains.append(
                next((k for k, ring in enumerate(rings, start=1) if point_in_polygon(centroid, ring)), 0)
            )
        return subdomains

    def _average_mesh_quality(self, mesh: Mesh) -> float:
        qualities = [triangle_quality(t.a, t.b, t.c) for t in mesh.triangles]
        return sum(qualities) / len(qualities)

    def _fallback_triangulate_front(
            self,
            front: list[FrontEdge],
            polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
    ) -> list[Triangle]:
        loop = self._order_front_loop(front)
        if loop is None or len(loop) < 3:
            return []
        return self._ear_clip_polygon(loop, polygon, holes, cut_segments)

    def _order_front_loop(self, front: list[FrontEdge]) -> list[Point] | None:
        if not front:
            return None

        next_by_start: dict[tuple[float, float], Point] = {}
        point_by_key: dict[tuple[float, float], Point] = {}
        for start, end in front:
            start_key = self._point_key(start)
            end_key = self._point_key(end)
            if start_key in next_by_start:
                return None
            next_by_start[start_key] = end
            point_by_key[start_key] = start
            point_by_key[end_key] = end

        start = front[0][0]
        start_key = self._point_key(start)
        loop: list[Point] = [start]
        current_key = start_key

        for _ in range(len(front)):
            next_point = next_by_start.get(current_key)
            if next_point is None:
                return None
            next_key = self._point_key(next_point)
            if next_key == start_key:
                break
            loop.append(next_point)
            current_key = next_key

        return loop if len(loop) >= 3 else None

    def _ear_clip_polygon(
            self,
            polygon: list[Point],
            domain_polygon: list[Point],
            holes: list[list[Point]],
            cut_segments: list[Segment],
    ) -> list[Triangle]:
        points = polygon.copy()
        triangles: list[Triangle] = []

        if self._signed_area(points) < 0:
            points.reverse()

        guard = 0
        while len(points) > 3 and guard < len(points) * len(points):
            ear_found = False
            n = len(points)
            for i in range(n):
                a = points[(i - 1) % n]
                b = points[i]
                c = points[(i + 1) % n]

                if orientation(a, b, c) <= 0:
                    continue
                if triangle_quality(a, b, c) < self._min_triangle_quality:
                    continue
                if not self._ear_has_reasonable_diagonal(a, b, c):
                    continue
                if not self._point_in_domain(Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0), domain_polygon, holes):
                    continue
                if not self._triangle_respects_holes(a, b, c, holes):
                    continue
                if not self._triangle_respects_cut_segments(a, b, c, cut_segments):
                    continue

                contains_point = False
                for j in range(n):
                    if j in ((i - 1) % n, i, (i + 1) % n):
                        continue
                    if self._point_in_triangle_strict(points[j], a, b, c):
                        contains_point = True
                        break
                if contains_point:
                    continue

                triangles.append(Triangle(a, b, c))
                points.pop(i)
                ear_found = True
                break

            if not ear_found:
                break
            guard += 1

        if len(points) == 3:
            a, b, c = points[0], points[1], points[2]
            if (
                    orientation(a, b, c) > 0
                    and self._ear_has_reasonable_diagonal(a, b, c)
                    and self._point_in_domain(Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0), domain_polygon, holes)
                    and self._triangle_respects_holes(a, b, c, holes)
                    and self._triangle_respects_cut_segments(a, b, c, cut_segments)
            ):
                triangles.append(Triangle(a, b, c))
        return triangles

    def _is_reasonable_closure_candidate(
            self,
            a: Point,
            b: Point,
            candidate: Point,
            base_len: float,
            max_ratio: float,
    ) -> bool:
        if base_len <= _EPSILON:
            return False
        ac = distance(a, candidate)
        bc = distance(b, candidate)
        limit = base_len * max_ratio
        return ac <= limit and bc <= limit

    def _ear_has_reasonable_diagonal(self, a: Point, b: Point, c: Point) -> bool:
        ab = distance(a, b)
        bc = distance(b, c)
        ac = distance(a, c)
        local_scale = max(ab, bc, _EPSILON)
        return ac <= local_scale * _MAX_EAR_DIAGONAL_FACTOR

    def _point_in_triangle_strict(self, p: Point, a: Point, b: Point, c: Point) -> bool:
        o1 = orientation(a, b, p)
        o2 = orientation(b, c, p)
        o3 = orientation(c, a, p)
        return o1 > 0 and o2 > 0 and o3 > 0

    def _build_topology(
            self,
            mesh: Mesh,
            boundary: list[Point],
            interface_points: list[Point] | None = None,
    ) -> tuple[
        dict[int, Point],
        list[TriangleIds],
        dict[int, set[int]],
        dict[int, list[int]],
        set[int],
        set[int],
    ]:
        linear_triangles = mesh.linear_triangles(key_precision=_KEY_PRECISION)
        positions: dict[int, Point] = {}
        adjacency: dict[int, set[int]] = {}
        node_triangles: dict[int, list[int]] = {}
        triangles: list[TriangleIds] = []
        key_to_id: dict[tuple[float, float], int] = {}

        for linear_triangle in linear_triangles:
            a_id, b_id, c_id = linear_triangle.node_numbers
            a, b, c = linear_triangle.node_coordinates
            triangles.append((a_id, b_id, c_id))

            if a_id not in positions:
                positions[a_id] = a
                key_to_id[self._point_key(a)] = a_id
                adjacency[a_id] = set()
                node_triangles[a_id] = []
            if b_id not in positions:
                positions[b_id] = b
                key_to_id[self._point_key(b)] = b_id
                adjacency[b_id] = set()
                node_triangles[b_id] = []
            if c_id not in positions:
                positions[c_id] = c
                key_to_id[self._point_key(c)] = c_id
                adjacency[c_id] = set()
                node_triangles[c_id] = []

            adjacency[a_id].update((b_id, c_id))
            adjacency[b_id].update((a_id, c_id))
            adjacency[c_id].update((a_id, b_id))

            tri_idx = linear_triangle.triangle_number - 1
            node_triangles[a_id].append(tri_idx)
            node_triangles[b_id].append(tri_idx)
            node_triangles[c_id].append(tri_idx)

        boundary_ids: set[int] = set()
        for point in boundary:
            key = self._point_key(point)
            node_id = key_to_id.get(key)
            if node_id is not None:
                boundary_ids.add(node_id)

        interface_ids: set[int] = set()
        for point in interface_points or []:
            key = self._point_key(point)
            node_id = key_to_id.get(key)
            if node_id is not None:
                interface_ids.add(node_id)

        return positions, triangles, adjacency, node_triangles, boundary_ids, interface_ids

    def _move_preserves_orientation(
            self,
            node_id: int,
            proposed: Point,
            positions: dict[int, Point],
            triangles: list[TriangleIds],
            node_triangles: dict[int, list[int]],
    ) -> bool:
        for tri_idx in node_triangles[node_id]:
            a_id, b_id, c_id = triangles[tri_idx]
            a = proposed if a_id == node_id else positions[a_id]
            b = proposed if b_id == node_id else positions[b_id]
            c = proposed if c_id == node_id else positions[c_id]

            if orientation(a, b, c) <= 0:
                return False
        return True

    def _point_key(self, point: Point) -> tuple[float, float]:
        return (round(point.x, _KEY_PRECISION), round(point.y, _KEY_PRECISION))

    def _normalize_cut_polyline(self, cut: list[Point], tolerance: float) -> list[Point]:
        if len(cut) < 2:
            return []
        cleaned: list[Point] = []
        for point in cut:
            if not cleaned or distance(cleaned[-1], point) > tolerance:
                cleaned.append(point)
        return cleaned if len(cleaned) >= 2 else []

    def _snap_point_to_vertices(self, point: Point, vertices: list[Point], tolerance: float) -> Point:
        for vertex in vertices:
            if distance(point, vertex) <= tolerance:
                return vertex
        return point

    def _validate_cut_polyline(
            self,
            cut: list[Point],
            boundary: list[Point],
            holes: list[list[Point]],
            tolerance: float,
    ) -> None:
        if len(cut) < 2:
            raise ValueError("Cut must contain at least 2 points.")

        for i in range(len(cut) - 1):
            a = cut[i]
            b = cut[i + 1]
            if distance(a, b) <= tolerance:
                raise ValueError("Cut contains too short segment.")
            midpoint = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
            if not point_in_polygon(midpoint, boundary, include_boundary=True):
                raise ValueError("Cut segment lies outside boundary.")
            if any(point_in_polygon(midpoint, hole, include_boundary=True) for hole in holes):
                raise ValueError("Cut intersects hole interior.")
            for hole in holes:
                for edge in self._polygon_edges(hole):
                    if segments_intersect(a, b, edge[0], edge[1], include_endpoints=True):
                        raise ValueError("Cut intersects or touches hole boundary.")

        self._ensure_cut_no_self_intersection(cut)

    def _ensure_cut_no_self_intersection(self, cut: list[Point]) -> None:
        edges = [(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]
        for i, first in enumerate(edges):
            for j in range(i + 1, len(edges)):
                if abs(i - j) <= 1:
                    continue
                second = edges[j]
                if segments_intersect(first[0], first[1], second[0], second[1], include_endpoints=True):
                    raise ValueError("Cut self-intersection is not allowed.")

    def _split_cut_segments(self, segments: list[Segment], tolerance: float) -> list[Segment]:
        if not segments:
            return []

        changed = True
        current = segments.copy()
        while changed:
            changed = False
            next_segments: list[Segment] = []
            used = [False] * len(current)
            for i, first in enumerate(current):
                if used[i]:
                    continue
                split_points_first: list[Point] = [first[0], first[1]]
                for j in range(i + 1, len(current)):
                    second = current[j]
                    intersection = self._segment_intersection_point(first[0], first[1], second[0], second[1], tolerance)
                    if intersection is None:
                        continue
                    if not self._point_is_endpoint(intersection, first, tolerance):
                        split_points_first.append(intersection)
                        changed = True
                    if not self._point_is_endpoint(intersection, second, tolerance):
                        used[j] = True
                        split_points_second = [second[0], intersection, second[1]]
                        next_segments.extend(self._segments_from_points(split_points_second, tolerance))
                        changed = True
                next_segments.extend(self._segments_from_points(split_points_first, tolerance))
            current = self._dedupe_segments(next_segments, tolerance)
        return current

    def _segment_intersection_point(
            self,
            a: Point,
            b: Point,
            c: Point,
            d: Point,
            tolerance: float,
    ) -> Point | None:
        if not segments_intersect(a, b, c, d, include_endpoints=True):
            return None

        denominator = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)
        if abs(denominator) <= tolerance:
            if self._share_endpoint((a, b), (c, d)):
                for point in (a, b):
                    if distance(point, c) <= tolerance or distance(point, d) <= tolerance:
                        return point
            raise ValueError("Overlapping collinear cut segments are not supported.")

        det1 = a.x * b.y - a.y * b.x
        det2 = c.x * d.y - c.y * d.x
        x = (det1 * (c.x - d.x) - (a.x - b.x) * det2) / denominator
        y = (det1 * (c.y - d.y) - (a.y - b.y) * det2) / denominator
        return Point(x, y)

    def _point_is_endpoint(self, point: Point, segment: Segment, tolerance: float) -> bool:
        return distance(point, segment[0]) <= tolerance or distance(point, segment[1]) <= tolerance

    def _segments_from_points(self, points: list[Point], tolerance: float) -> list[Segment]:
        unique: list[Point] = []
        for point in points:
            if not unique or distance(unique[-1], point) > tolerance:
                unique.append(point)
        output: list[Segment] = []
        for i in range(len(unique) - 1):
            if distance(unique[i], unique[i + 1]) > tolerance:
                output.append((unique[i], unique[i + 1]))
        return output

    def _dedupe_segments(self, segments: list[Segment], tolerance: float) -> list[Segment]:
        deduped: list[Segment] = []
        for segment in segments:
            if distance(segment[0], segment[1]) <= tolerance:
                continue
            if any(self._same_undirected_edge(segment, existing) for existing in deduped):
                continue
            deduped.append(segment)
        return deduped

    def _validate_cut_segments_are_mesh_edges(self, mesh: Mesh, cut_segments: list[Segment]) -> None:
        if not cut_segments:
            return
        mesh_edges = {
            self._undirected_edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }
        for segment in cut_segments:
            if self._undirected_edge_key(segment) not in mesh_edges:
                raise ValueError("AFM failed to preserve all constrained cut segments in the mesh.")

    def _undirected_edge_key(self, edge: Segment) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = edge
        return tuple(sorted((self._point_key(a), self._point_key(b))))

    def _subdivide_cut_segments(self, cut_segments: list[Segment], h: float) -> list[Segment]:
        if not cut_segments:
            return []
        subdivided: list[Segment] = []
        for a, b in cut_segments:
            seg_len = distance(a, b)
            if seg_len <= _EPSILON:
                continue
            k = max(1, ceil(seg_len / h))
            prev = a
            for j in range(1, k + 1):
                t = j / k
                curr = Point(
                    a.x + (b.x - a.x) * t,
                    a.y + (b.y - a.y) * t,
                    )
                if distance(prev, curr) > _EPSILON:
                    subdivided.append((prev, curr))
                prev = curr
        return self._dedupe_segments(subdivided, tolerance=max(_EPSILON * 100.0, 1e-6))

    def _insert_points_on_edges(self, polygon: list[Point], points: list[Point]) -> list[Point]:
        result: list[Point] = []
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            result.append(a)
            inner = {p for p in points if p != a and p != b and self._point_inside_segment(p, a, b)}
            result.extend(sorted(inner, key=lambda p: distance(a, p)))
        return result

    def _nearest_front_vertex(
            self,
            point: Point,
            radius: float,
            exclude: tuple[Point, Point],
            front: list[FrontEdge],
            spatial_index: dict[tuple[int, int], list[int]] | None,
            cell_size: float | None,
    ) -> Point | None:
        if spatial_index is not None and cell_size is not None and cell_size > _EPSILON:
            edge_ids = self._query_segment_spatial_index(
                spatial_index,
                Point(point.x - radius, point.y - radius),
                Point(point.x + radius, point.y + radius),
                cell_size,
            )
            vertices = {v for idx in edge_ids for v in front[idx]}
        else:
            vertices = {v for edge in front for v in edge}
        best: Point | None = None
        best_distance = radius
        for v in vertices:
            if v in exclude:
                continue
            d = distance(point, v)
            if d < best_distance:
                best, best_distance = v, d
        return best

    def _improve_topology(self, mesh: Mesh, fixed_points: list[Point], cut_segments: list[Segment]) -> Mesh:
        """Edge flips plus removal of interior valence-3 nodes. Boundary, hole and cut nodes stay;
        cut edges are never flipped."""
        key_to_id: dict[tuple[float, float], int] = {}
        nodes: list[tuple[float, float]] = []

        def node_id(point: Point) -> int:
            key = self._point_key(point)
            if key not in key_to_id:
                key_to_id[key] = len(nodes)
                nodes.append((point.x, point.y))
            return key_to_id[key]

        triangles = [(node_id(t.a), node_id(t.b), node_id(t.c)) for t in mesh.triangles]
        fixed = {key_to_id[key] for key in (self._point_key(p) for p in fixed_points) if key in key_to_id}
        locked: set[tuple[int, int]] = set()
        for a, b in cut_segments:
            ka, kb = self._point_key(a), self._point_key(b)
            if ka in key_to_id and kb in key_to_id:
                locked.add(tuple(sorted((key_to_id[ka], key_to_id[kb]))))

        removed_total = 0
        for _ in range(_TOPOLOGY_ROUNDS):
            triangles = improve_mesh_by_edge_flips(nodes, triangles, max_passes=3, locked_edges=locked)
            triangles, removed = self._remove_valence3_nodes(nodes, triangles, fixed)
            removed_total += removed
            if not removed:
                break
        _afm_debug("afm.topology.improved", removed_valence3=removed_total, triangles=len(triangles))
        points = [Point(x, y) for x, y in nodes]
        return Mesh(
            triangles=[Triangle(points[a], points[b], points[c]) for a, b, c in triangles],
            meters_per_pixel=mesh.meters_per_pixel,
        )

    def _remove_valence3_nodes(
            self,
            nodes: list[tuple[float, float]],
            triangles: list[TriangleIds],
            fixed: set[int],
    ) -> tuple[list[TriangleIds], int]:
        incident: dict[int, list[int]] = defaultdict(list)
        for idx, triangle in enumerate(triangles):
            for n in triangle:
                incident[n].append(idx)
        consumed: set[int] = set()
        merged: list[TriangleIds] = []
        min_quality = max(self._min_triangle_quality, _VALENCE3_MERGE_MIN_QUALITY)
        for n, tri_ids in incident.items():
            if n in fixed or len(tri_ids) != 3 or any(i in consumed for i in tri_ids):
                continue
            ring = sorted({m for i in tri_ids for m in triangles[i]} - {n})
            if len(ring) != 3:
                continue
            a, b, c = (Point(*nodes[m]) for m in ring)
            if orientation(a, b, c) < 0:
                ring = [ring[0], ring[2], ring[1]]
                a, b, c = (Point(*nodes[m]) for m in ring)
            if not self._point_in_triangle_strict(Point(*nodes[n]), a, b, c):
                continue
            try:
                if triangle_quality(a, b, c) < min_quality:
                    continue
            except ValueError:
                continue
            consumed.update(tri_ids)
            merged.append((ring[0], ring[1], ring[2]))
        if not merged:
            return triangles, 0
        kept = [t for i, t in enumerate(triangles) if i not in consumed]
        return kept + merged, len(merged)

    def _check_deadline(self) -> None:
        if self._deadline is not None and monotonic() > self._deadline:
            _afm_debug("afm.time_budget_exceeded", time_budget_s=self._time_budget_s)
            raise MeshingTimeoutError(
                "Тріангуляція перевищила ліміт часу. Спробуйте інший режим (наприклад, Fast) "
                "або спростіть контури."
            )

    def _domain_area(self, polygon: list[Point], holes: list[list[Point]]) -> float:
        return abs(self._signed_area(polygon)) - sum(abs(self._signed_area(hole)) for hole in holes)

    def _overlaps_built_triangles(self, a: Point, b: Point, c: Point) -> bool:
        index = self._built_index
        if index is None:
            return False
        candidate = (a, b, c)
        return any(self._triangles_conflict(candidate, index.triangles[i]) for i in index.query(a, b, c))

    def _validate_final_mesh(self, mesh: Mesh, domain_area: float) -> None:
        """Raise ValueError unless the triangles tile the domain: total area matches, interiors do
        not overlap and no vertex lies inside another triangle's side (hanging node)."""
        triangles = [(t.a, t.b, t.c) if orientation(t.a, t.b, t.c) > 0 else (t.a, t.c, t.b) for t in mesh.triangles]
        total_area = sum(triangle_area(*t) for t in triangles)
        if abs(total_area - domain_area) > _AREA_RELATIVE_TOLERANCE * max(domain_area, 1.0):
            raise ValueError(
                f"Mesh validation failed: triangle area {total_area:.6f} != domain area {domain_area:.6f}."
            )
        longest = max(distance(p, q) for t in triangles for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
        index = _TriangleSpatialIndex(cell_size=longest)
        for i, triangle in enumerate(triangles):
            for j in index.query(*triangle):
                if self._triangles_conflict(triangle, index.triangles[j]):
                    raise ValueError(f"Mesh validation failed: triangles {j} and {i} overlap or form a hanging node.")
            index.insert(*triangle)

    def _triangles_conflict(
            self,
            first: tuple[Point, Point, Point],
            second: tuple[Point, Point, Point],
    ) -> bool:
        """True when two CCW triangles overlap in their interiors, or a vertex of one lies strictly
        inside the other or inside one of its sides. Sharing a vertex or an edge is fine."""
        if (
                max(p.x for p in first) < min(p.x for p in second)
                or max(p.x for p in second) < min(p.x for p in first)
                or max(p.y for p in first) < min(p.y for p in second)
                or max(p.y for p in second) < min(p.y for p in first)
        ):
            return False
        shared = [p for p in first if p in second]
        if len(shared) == 3:
            return True
        if len(shared) == 2:
            u, v = shared
            third_first = next(p for p in first if p not in shared)
            third_second = next(p for p in second if p not in shared)
            if orientation(u, v, third_first) * orientation(u, v, third_second) >= 0:
                return True
        for this, other in ((first, second), (second, first)):
            for p in this:
                if p in other:
                    continue
                if self._point_in_triangle_strict(p, *other):
                    return True
                for q, r in ((other[0], other[1]), (other[1], other[2]), (other[2], other[0])):
                    if self._point_inside_segment(p, q, r):
                        return True
        for p1, p2 in ((first[0], first[1]), (first[1], first[2]), (first[2], first[0])):
            for q1, q2 in ((second[0], second[1]), (second[1], second[2]), (second[2], second[0])):
                if self._share_endpoint((p1, p2), (q1, q2)):
                    continue
                if segments_intersect(p1, p2, q1, q2, include_endpoints=False):
                    return True
        return False

    def _point_inside_segment(self, p: Point, a: Point, b: Point) -> bool:
        dx = b.x - a.x
        dy = b.y - a.y
        length_sq = dx * dx + dy * dy
        if length_sq <= _EPSILON:
            return False
        t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / length_sq
        if t <= 1e-9 or t >= 1.0 - 1e-9:
            return False
        length = sqrt(length_sq)
        line_distance = abs(dx * (p.y - a.y) - dy * (p.x - a.x)) / length
        return line_distance <= _ON_EDGE_RELATIVE_TOLERANCE * max(length, 1.0)

    def _signed_area(self, polygon: list[Point]) -> float:
        area = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y
        return area / 2.0
