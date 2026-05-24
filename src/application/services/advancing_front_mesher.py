from __future__ import annotations

from math import ceil, sqrt

from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import (
    distance,
    orientation,
    point_in_polygon,
    segments_intersect,
    triangle_quality,
)
from src.domain.interfaces.imesh_generator import IMeshGenerator

FrontEdge = tuple[Point, Point]
TriangleIds = tuple[int, int, int]

_EPSILON = 1e-9
_KEY_PRECISION = 10


class AdvancingFrontMesher(IMeshGenerator):
    def __init__(
        self,
        min_triangle_quality: float = 0.02,
        max_iterations_factor: int = 500,
        target_edge_length: float | None = None,
        smoothing_iterations: int = 5,
    ) -> None:
        if not 0.0 <= min_triangle_quality <= 1.0:
            raise ValueError("min_triangle_quality must be in [0, 1]")
        if max_iterations_factor < 1:
            raise ValueError("max_iterations_factor must be >= 1")
        if target_edge_length is not None and target_edge_length <= 0:
            raise ValueError("target_edge_length must be > 0 when provided")
        if not 0 <= smoothing_iterations <= 20:
            raise ValueError("smoothing_iterations must be in [0, 20]")

        self._min_triangle_quality = min_triangle_quality
        self._max_iterations_factor = max_iterations_factor
        self._target_edge_length = target_edge_length
        self._smoothing_iterations = smoothing_iterations
        self._steiner_activation_length_factor = 0.95

    def generate(self, boundary: list[Point]) -> Mesh:
        return self.generate_with_holes(boundary, holes=[])

    def generate_with_holes(self, boundary: list[Point], holes: list[list[Point]]) -> Mesh:
        return self.generate_with_holes_and_cuts(boundary, holes, cuts=[])

    def generate_with_holes_and_cuts(
        self,
        boundary: list[Point],
        holes: list[list[Point]],
        cuts: list[list[Point]],
    ) -> Mesh:
        polygon = self._prepare_boundary(boundary)
        hole_polygons = [self._prepare_hole_boundary(hole) for hole in holes if hole]
        cut_polygons = [self._prepare_cut_boundary(cut, polygon) for cut in cuts if len(cut) >= 2]
        base_h = self._resolve_target_step(polygon)

        best_mesh: Mesh | None = None
        best_quality = -1.0
        quality_target = 0.7

        for attempt in range(4):
            target_h = base_h * (0.8 ** attempt)
            mesh, fixed_boundaries = self._generate_single_pass(polygon, hole_polygons, cut_polygons, target_h)
            mesh = self.smooth(mesh, boundary=fixed_boundaries, iterations=self._smoothing_iterations)
            avg_quality = self._average_mesh_quality(mesh)

            if avg_quality > best_quality:
                best_quality = avg_quality
                best_mesh = mesh
            if avg_quality >= quality_target:
                return mesh

        if best_mesh is None:
            raise ValueError("AFM failed to generate mesh.")
        return best_mesh

    def _generate_single_pass(
        self,
        polygon: list[Point],
        holes: list[list[Point]],
        cuts: list[list[Point]],
        target_h: float,
    ) -> tuple[Mesh, list[Point]]:
        polygon = self._subdivide_boundary(polygon, target_h)
        holes = [self._subdivide_boundary(hole, target_h) for hole in holes]
        cuts = [self._subdivide_boundary(cut, target_h) for cut in cuts]
        front = self._build_initial_front(polygon)
        for hole in holes:
            front.extend(self._build_initial_front(hole))
        for cut in cuts:
            front.extend(self._build_initial_front(cut))
        triangles: list[Triangle] = []

        max_iterations = max(200, len(front) * self._max_iterations_factor)
        iterations = 0

        while front and iterations < max_iterations:
            advancement = self._find_advancement(front, polygon, holes, cuts, target_h)
            if advancement is None:
                fallback_triangles = self._fallback_triangulate_front(front)
                if not fallback_triangles:
                    raise ValueError("AFM stalled: active front cannot be advanced further.")
                triangles.extend(fallback_triangles)
                front.clear()
                break

            active_idx, a, b, candidate = advancement
            front.pop(active_idx)

            triangles.append(Triangle(a, b, candidate))
            self._update_front(front, (a, candidate))
            self._update_front(front, (candidate, b))
            iterations += 1

        if front:
            raise ValueError("AFM failed to close the front before iteration limit.")
        if not triangles:
            raise ValueError("AFM failed to generate mesh.")

        fixed_boundary = [*polygon]
        for hole in holes:
            fixed_boundary.extend(hole)
        for cut in cuts:
            fixed_boundary.extend(cut)
        return Mesh(triangles=triangles), fixed_boundary

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

    def _prepare_cut_boundary(self, cut: list[Point], boundary: list[Point]) -> list[Point]:
        if len(cut) < 2:
            raise ValueError("Cut must contain at least 2 points.")

        cleaned: list[Point] = []
        for point in cut:
            if not cleaned or distance(point, cleaned[-1]) > _EPSILON:
                cleaned.append(point)
        if len(cleaned) < 2:
            raise ValueError("Cut must contain at least 2 distinct points.")

        target_h = self._resolve_target_step(boundary)
        # Represent a cut as a thin closed barrier so the front grows on both sides of the line.
        thickness = max(min(target_h * 0.05, 0.75), 0.35)
        half_thickness = thickness / 2.0

        left_side: list[Point] = []
        right_side: list[Point] = []
        for idx, point in enumerate(cleaned):
            normal = self._cut_vertex_normal(cleaned, idx)
            left_side.append(Point(point.x + normal.x * half_thickness, point.y + normal.y * half_thickness))
            right_side.append(Point(point.x - normal.x * half_thickness, point.y - normal.y * half_thickness))

        polygon = [*left_side, *reversed(right_side)]
        prepared = self._prepare_boundary(polygon)
        if self._signed_area(prepared) > 0:
            prepared.reverse()
        return prepared

    def _resolve_target_step(self, boundary: list[Point]) -> float:
        if self._target_edge_length is not None:
            return self._target_edge_length

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

    def _sorted_front_indices_by_priority(self, front: list[FrontEdge]) -> list[int]:
        return sorted(range(len(front)), key=lambda i: distance(front[i][0], front[i][1]))

    def _find_advancement(
        self,
        front: list[FrontEdge],
        polygon: list[Point],
        holes: list[list[Point]],
        cuts: list[list[Point]],
        target_step: float,
    ) -> tuple[int, Point, Point, Point] | None:
        for idx in self._sorted_front_indices_by_priority(front):
            a, b = front[idx]
            other_edges = front[:idx] + front[idx + 1:]
            candidate = self._find_best_node(a, b, polygon, holes, cuts, other_edges, target_step)
            if candidate is not None:
                return idx, a, b, candidate
        return None

    def _find_best_node(
        self,
        a: Point,
        b: Point,
        polygon: list[Point],
        holes: list[list[Point]],
        cuts: list[list[Point]],
        front: list[FrontEdge],
        target_step: float,
    ) -> Point | None:
        edge_len = distance(a, b)
        if edge_len <= _EPSILON:
            return None

        midpoint = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
        existing_vertices = self._front_vertices(front, exclude={a, b})

        # 1) Prefer interior growth via Steiner points.
        if edge_len > target_step * self._steiner_activation_length_factor:
            for p in self._steiner_candidates(a, b):
                if self._is_too_close_to_existing_edges(p, front, threshold=0.4 * target_step):
                    continue
                if self._is_valid_triangle(a, b, p, polygon, holes, cuts, front):
                    return p

        # 2) Try to close topology with existing nodes within local radius L.
        near_vertices = [p for p in existing_vertices if distance(p, midpoint) <= edge_len + _EPSILON]
        near_vertices.sort(key=lambda p: distance(p, midpoint))
        for p in near_vertices:
            if self._is_valid_triangle(a, b, p, polygon, holes, cuts, front):
                return p

        # 3) Wider closure radius for final wave connection.
        closure_radius = edge_len * 2.5
        far_vertices = [p for p in existing_vertices if distance(p, midpoint) <= closure_radius]
        far_vertices.sort(key=lambda p: distance(p, midpoint))
        for p in far_vertices:
            if self._is_valid_triangle(a, b, p, polygon, holes, cuts, front):
                return p

        return None

    def _is_too_close_to_existing_edges(self, point: Point, edges: list[FrontEdge], threshold: float) -> bool:
        for edge in edges:
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
        for h_scale in (1.0, 0.75, 0.5):
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
        cuts: list[list[Point]],
        front: list[FrontEdge],
    ) -> bool:
        if c == a or c == b:
            return False
        if orientation(a, b, c) <= 0:
            return False
        if not self._point_in_domain(c, polygon, holes, cuts):
            return False

        centroid = Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
        if not self._point_in_domain(centroid, polygon, holes, cuts):
            return False
        if not self._triangle_respects_obstacles(a, b, c, holes, cuts):
            return False

        try:
            quality = triangle_quality(a, b, c)
        except ValueError:
            return False
        if quality < self._min_triangle_quality:
            return False

        for new_edge in ((a, c), (c, b)):
            if self._edge_intersects_front(new_edge, front):
                return False

        if self._contains_front_vertex(a, b, c, front):
            return False
        return True

    def _point_in_domain(
        self,
        point: Point,
        polygon: list[Point],
        holes: list[list[Point]],
        cuts: list[list[Point]],
    ) -> bool:
        if not point_in_polygon(point, polygon, include_boundary=True):
            return False
        if any(point_in_polygon(point, hole, include_boundary=False) for hole in holes):
            return False
        return not any(point_in_polygon(point, cut, include_boundary=False) for cut in cuts)

    def _triangle_respects_obstacles(
        self,
        a: Point,
        b: Point,
        c: Point,
        holes: list[list[Point]],
        cuts: list[list[Point]],
    ) -> bool:
        triangle_edges = ((a, b), (b, c), (c, a))
        for obstacle in (*holes, *cuts):
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

    def _polygon_edges(self, polygon: list[Point]) -> list[FrontEdge]:
        return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]

    def _edge_intersects_front(self, edge: FrontEdge, front: list[FrontEdge]) -> bool:
        for current in front:
            if not segments_intersect(edge[0], edge[1], current[0], current[1], include_endpoints=False):
                continue
            if self._share_endpoint(edge, current):
                continue
            return True
        return False

    def _contains_front_vertex(self, a: Point, b: Point, c: Point, front: list[FrontEdge]) -> bool:
        for p in self._front_vertices(front, exclude={a, b, c}):
            o1 = orientation(a, b, p)
            o2 = orientation(b, c, p)
            o3 = orientation(c, a, p)
            if o1 > 0 and o2 > 0 and o3 > 0:
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

    def _share_endpoint(self, e1: FrontEdge, e2: FrontEdge) -> bool:
        return e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]

    def smooth(self, mesh: Mesh, boundary: list[Point], iterations: int = 5) -> Mesh:
        return self._laplacian_smooth(mesh, boundary, iterations)

    def _laplacian_smooth(self, mesh: Mesh, boundary: list[Point], iterations: int) -> Mesh:
        node_positions, triangles, adjacency, node_triangles, boundary_ids = self._build_topology(mesh, boundary)
        if not node_positions:
            return mesh

        internal_ids = [node_id for node_id in node_positions if node_id not in boundary_ids]
        positions = dict(node_positions)

        for _ in range(iterations):
            proposed_positions = dict(positions)
            for node_id in internal_ids:
                neighbors = adjacency[node_id]
                if not neighbors:
                    continue

                avg_x = sum(positions[n].x for n in neighbors) / len(neighbors)
                avg_y = sum(positions[n].y for n in neighbors) / len(neighbors)
                proposed = Point(avg_x, avg_y)

                if self._move_preserves_orientation(
                    node_id=node_id,
                    proposed=proposed,
                    positions=proposed_positions,
                    triangles=triangles,
                    node_triangles=node_triangles,
                ):
                    proposed_positions[node_id] = proposed
            positions = proposed_positions

        smoothed_triangles = [
            Triangle(positions[a], positions[b], positions[c])
            for a, b, c in triangles
        ]
        return Mesh(triangles=smoothed_triangles)

    def _average_mesh_quality(self, mesh: Mesh) -> float:
        qualities = [triangle_quality(t.a, t.b, t.c) for t in mesh.triangles]
        return sum(qualities) / len(qualities)

    def _fallback_triangulate_front(self, front: list[FrontEdge]) -> list[Triangle]:
        loop = self._order_front_loop(front)
        if loop is None or len(loop) < 3:
            return []
        return self._ear_clip_polygon(loop)

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

    def _ear_clip_polygon(self, polygon: list[Point]) -> list[Triangle]:
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
            triangles.append(Triangle(points[0], points[1], points[2]))
        return triangles

    def _point_in_triangle_strict(self, p: Point, a: Point, b: Point, c: Point) -> bool:
        o1 = orientation(a, b, p)
        o2 = orientation(b, c, p)
        o3 = orientation(c, a, p)
        return o1 > 0 and o2 > 0 and o3 > 0

    def _build_topology(
        self,
        mesh: Mesh,
        boundary: list[Point],
    ) -> tuple[
        dict[int, Point],
        list[TriangleIds],
        dict[int, set[int]],
        dict[int, list[int]],
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

        return positions, triangles, adjacency, node_triangles, boundary_ids

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

    def _signed_area(self, polygon: list[Point]) -> float:
        area = 0.0
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y
        return area / 2.0
