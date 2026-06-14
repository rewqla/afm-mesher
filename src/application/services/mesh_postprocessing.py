from __future__ import annotations

from collections import deque

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via explicit error path
    np = None  # type: ignore[assignment]

from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import orientation, triangle_quality

Node = tuple[float, float]
TriangleIndices = tuple[int, int, int]

_TRIANGLE_KEY_NODE_BITS = 32


def compute_max_difference(triangles: list[LinearTriangle]) -> int:
    """Return the maximum node-number spread inside any triangle."""
    return _compute_bandwidth([triangle.node_numbers for triangle in triangles])


def renumber_triangles(
    triangles: list[LinearTriangle],
) -> list[LinearTriangle]:
    """Return a new list of LinearTriangles with stable 1-based numbering."""
    if np is None:
        raise ImportError("numpy is required for renumber_triangles")
    if not triangles:
        return []
    if len(triangles) == 1:
        triangle = triangles[0]
        return [
            LinearTriangle(
                triangle_number=1,
                node_coordinates=triangle.node_coordinates,
                node_numbers=triangle.node_numbers,
                meters_per_pixel=triangle.meters_per_pixel,
            )
        ]

    keys = np.array(
        [
            (min(triangle.node_numbers) << _TRIANGLE_KEY_NODE_BITS) | max(triangle.node_numbers)
            for triangle in triangles
        ],
        dtype=np.uint64,
    )
    order = np.argsort(keys, kind="stable")

    return [
        LinearTriangle(
            triangle_number=new_number,
            node_coordinates=triangles[int(index)].node_coordinates,
            node_numbers=triangles[int(index)].node_numbers,
            meters_per_pixel=triangles[int(index)].meters_per_pixel,
        )
        for new_number, index in enumerate(order, start=1)
    ]


def renumber_nodes_rcm(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    if not nodes:
        return [], [], 0
    if not triangles:
        return list(nodes), [], 0

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    degrees = [len(neighbors) for neighbors in adjacency]

    bfs_order: list[int] = []
    visited = [False] * len(nodes)

    while len(bfs_order) < len(nodes):
        remaining = [node_id for node_id in range(len(nodes)) if not visited[node_id]]
        start = min(remaining, key=lambda node_id: (degrees[node_id], node_id))
        queue: deque[int] = deque([start])
        visited[start] = True

        while queue:
            current = queue.popleft()
            bfs_order.append(current)

            next_nodes = [neighbor for neighbor in adjacency[current] if not visited[neighbor]]
            next_nodes.sort(key=lambda node_id: (degrees[node_id], node_id))
            for neighbor in next_nodes:
                visited[neighbor] = True
                queue.append(neighbor)

    rcm_order = list(reversed(bfs_order))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(rcm_order)}

    new_nodes = [nodes[old_id] for old_id in rcm_order]
    remapped_triangles_zero_based = [
        tuple(old_to_new[node_id] for node_id in triangle)
        for triangle in normalized_triangles
    ]
    bandwidth = _compute_bandwidth(remapped_triangles_zero_based)
    new_triangles = _restore_triangle_indices(remapped_triangles_zero_based, index_base)
    return new_nodes, new_triangles, bandwidth


def laplacian_smooth(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    boundary_nodes: set[int],
    iterations: int = 10,
    alpha: float = 0.5,
    tolerance: float = 1e-6,
) -> list[tuple[float, float]]:
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    if not nodes:
        return []
    if not triangles:
        return list(nodes)

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    normalized_boundary = _normalize_boundary_indices(boundary_nodes, len(nodes), index_base)
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    incident_triangles = _build_incident_triangles(len(nodes), normalized_triangles)

    positions = [Point(float(x), float(y)) for x, y in nodes]
    internal_nodes = [node_id for node_id in range(len(nodes)) if node_id not in normalized_boundary]

    for _ in range(iterations):
        proposed_positions = list(positions)
        max_displacement = 0.0

        for node_id in internal_nodes:
            neighbors = sorted(adjacency[node_id])
            if not neighbors:
                continue

            avg_x = sum(positions[neighbor].x for neighbor in neighbors) / len(neighbors)
            avg_y = sum(positions[neighbor].y for neighbor in neighbors) / len(neighbors)
            current = positions[node_id]
            accepted = False
            trial_alpha = alpha

            for _attempt in range(4):
                candidate = Point(
                    current.x + (avg_x - current.x) * trial_alpha,
                    current.y + (avg_y - current.y) * trial_alpha,
                )

                if _move_improves_local_quality(
                    node_id=node_id,
                    candidate=candidate,
                    positions=proposed_positions,
                    triangles=normalized_triangles,
                    incident_triangles=incident_triangles,
                    tolerance=tolerance,
                ):
                    proposed_positions[node_id] = candidate
                    dx = candidate.x - current.x
                    dy = candidate.y - current.y
                    max_displacement = max(max_displacement, (dx * dx + dy * dy) ** 0.5)
                    accepted = True
                    break
                trial_alpha *= 0.5

            if not accepted:
                continue

        positions = proposed_positions
        if max_displacement <= tolerance:
            break

    return [(point.x, point.y) for point in positions]


def improve_mesh_by_edge_flips(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    max_passes: int = 3,
    tolerance: float = 1e-9,
) -> list[tuple[int, int, int]]:
    if max_passes < 0:
        raise ValueError("max_passes must be >= 0")
    if not nodes or not triangles or max_passes == 0:
        return list(triangles)

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    positions = [Point(float(x), float(y)) for x, y in nodes]
    current = list(normalized_triangles)

    for _ in range(max_passes):
        changed = False
        edge_to_triangles = _build_edge_to_triangles(current)
        locked_triangles: set[int] = set()

        for edge, triangle_indices in edge_to_triangles.items():
            if len(triangle_indices) != 2:
                continue

            first_idx, second_idx = triangle_indices
            if first_idx in locked_triangles or second_idx in locked_triangles:
                continue

            first = current[first_idx]
            second = current[second_idx]
            shared = set(edge)
            first_other = next(node_id for node_id in first if node_id not in shared)
            second_other = next(node_id for node_id in second if node_id not in shared)

            if first_other == second_other:
                continue

            old_quality = _triangle_quality_tuple(first, positions, tolerance) + _triangle_quality_tuple(second, positions, tolerance)
            candidate_first = _orient_triangle(first_other, second_other, edge[0], positions, tolerance)
            candidate_second = _orient_triangle(second_other, first_other, edge[1], positions, tolerance)
            if candidate_first is None or candidate_second is None:
                continue
            if len({*candidate_first}) < 3 or len({*candidate_second}) < 3:
                continue

            new_quality = _triangle_quality_tuple(candidate_first, positions, tolerance) + _triangle_quality_tuple(candidate_second, positions, tolerance)
            if not _quality_tuple_is_better(new_quality, old_quality, tolerance):
                continue

            current[first_idx] = candidate_first
            current[second_idx] = candidate_second
            locked_triangles.update((first_idx, second_idx))
            changed = True

        if not changed:
            break

    return _restore_triangle_indices(current, index_base)


def _normalize_triangle_indices(
    triangles: list[tuple[int, int, int]],
    node_count: int,
) -> tuple[list[TriangleIndices], int]:
    flat_indices = [node_id for triangle in triangles for node_id in triangle]
    if any(not isinstance(node_id, int) for node_id in flat_indices):
        raise ValueError("triangle indices must be integers")

    min_index = min(flat_indices)
    max_index = max(flat_indices)

    if min_index >= 0 and max_index < node_count:
        return [tuple(triangle) for triangle in triangles], 0
    if min_index >= 1 and max_index <= node_count:
        return [tuple(node_id - 1 for node_id in triangle) for triangle in triangles], 1

    raise ValueError("triangle indices are out of bounds for the provided nodes")


def _normalize_boundary_indices(
    boundary_nodes: set[int],
    node_count: int,
    index_base: int,
) -> set[int]:
    normalized: set[int] = set()
    for node_id in boundary_nodes:
        if not isinstance(node_id, int):
            raise ValueError("boundary node indices must be integers")
        normalized_id = node_id - index_base
        if not 0 <= normalized_id < node_count:
            raise ValueError("boundary node indices are out of bounds for the provided nodes")
        normalized.add(normalized_id)
    return normalized


def _restore_triangle_indices(
    triangles: list[TriangleIndices],
    index_base: int,
) -> list[TriangleIndices]:
    if index_base == 0:
        return triangles
    return [tuple(node_id + index_base for node_id in triangle) for triangle in triangles]


def _build_node_adjacency(node_count: int, triangles: list[TriangleIndices]) -> list[set[int]]:
    adjacency = [set() for _ in range(node_count)]
    for a, b, c in triangles:
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def _build_incident_triangles(node_count: int, triangles: list[TriangleIndices]) -> list[list[int]]:
    incident = [[] for _ in range(node_count)]
    for triangle_index, triangle in enumerate(triangles):
        for node_id in triangle:
            incident[node_id].append(triangle_index)
    return incident


def _compute_bandwidth(triangles: list[TriangleIndices]) -> int:
    return max((_triangle_difference(triangle) for triangle in triangles), default=0)


def _triangle_difference(triangle: TriangleIndices) -> int:
    return max(triangle) - min(triangle)


def _move_improves_local_quality(
    node_id: int,
    candidate: Point,
    positions: list[Point],
    triangles: list[TriangleIndices],
    incident_triangles: list[list[int]],
    tolerance: float,
) -> bool:
    before = _local_quality_tuple(node_id, positions[node_id], positions, triangles, incident_triangles, tolerance)
    after = _local_quality_tuple(node_id, candidate, positions, triangles, incident_triangles, tolerance)
    return _quality_tuple_is_better(after, before, tolerance)


def _local_quality_tuple(
    node_id: int,
    candidate: Point,
    positions: list[Point],
    triangles: list[TriangleIndices],
    incident_triangles: list[list[int]],
    tolerance: float,
) -> tuple[float, float]:
    qualities: list[float] = []
    for triangle_index in incident_triangles[node_id]:
        a_id, b_id, c_id = triangles[triangle_index]
        a = candidate if a_id == node_id else positions[a_id]
        b = candidate if b_id == node_id else positions[b_id]
        c = candidate if c_id == node_id else positions[c_id]
        if orientation(a, b, c) <= 0:
            return (-1.0, -1.0)
        try:
            qualities.append(triangle_quality(a, b, c))
        except ValueError:
            return (-1.0, -1.0)
    if not qualities:
        return (-1.0, -1.0)
    return (min(qualities), sum(qualities) / len(qualities))


def _quality_tuple_is_better(
    candidate: tuple[float, float],
    baseline: tuple[float, float],
    tolerance: float,
) -> bool:
    if candidate[0] > baseline[0] + tolerance:
        return True
    if abs(candidate[0] - baseline[0]) <= tolerance and candidate[1] > baseline[1] + tolerance:
        return True
    return False


def _build_edge_to_triangles(
    triangles: list[TriangleIndices],
) -> dict[tuple[int, int], list[int]]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_to_triangles.setdefault(key, []).append(triangle_index)
    return edge_to_triangles


def _triangle_quality_tuple(
    triangle: TriangleIndices,
    positions: list[Point],
    tolerance: float,
) -> tuple[float, float]:
    a = positions[triangle[0]]
    b = positions[triangle[1]]
    c = positions[triangle[2]]
    if orientation(a, b, c) <= 0:
        return (-1.0, -1.0)
    try:
        quality = triangle_quality(a, b, c)
    except ValueError:
        return (-1.0, -1.0)
    return (quality, quality)


def _orient_triangle(
    a_id: int,
    b_id: int,
    c_id: int,
    positions: list[Point],
    tolerance: float,
) -> TriangleIndices | None:
    a = positions[a_id]
    b = positions[b_id]
    c = positions[c_id]
    turn = orientation(a, b, c)
    if turn == 0:
        return None
    if turn < 0:
        return (a_id, c_id, b_id)
    return (a_id, b_id, c_id)
