from __future__ import annotations

import heapq
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
    start_node = _select_min_degree_start(degrees)

    return _renumber_nodes_rcm_from_start(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        index_base=index_base,
        adjacency=adjacency,
        degrees=degrees,
        start_node=start_node,
    )


def renumber_nodes_rcm_multistart(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    candidates: list[int] | None = None,
    max_candidates: int = 20,
    probe_stride: int = 12,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    if not nodes:
        return [], [], 0
    if not triangles:
        return list(nodes), [], 0
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    degrees = [len(neighbors) for neighbors in adjacency]
    base_candidate_cap = min(max_candidates, 15)
    candidate_nodes = _normalize_rcm_candidates(candidates, degrees, max_candidates=base_candidate_cap)
    if candidates is None:
        candidate_nodes = _expand_rcm_candidate_pool(
            adjacency=adjacency,
            degrees=degrees,
            candidate_nodes=candidate_nodes,
            max_candidates=max_candidates,
            probe_stride=probe_stride,
        )

    first_round_results = _evaluate_rcm_candidates(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        index_base=index_base,
        adjacency=adjacency,
        degrees=degrees,
        start_nodes=candidate_nodes,
    )
    best_result = min(first_round_results, key=lambda result: (result[2], result[3]))

    if candidates is None:
        second_round_starts = _expand_rcm_refinement_starts(
            adjacency=adjacency,
            degrees=degrees,
            first_round_results=first_round_results,
            max_candidates=max_candidates,
        )
        if second_round_starts:
            second_round_results = _evaluate_rcm_candidates(
                nodes=nodes,
                normalized_triangles=normalized_triangles,
                index_base=index_base,
                adjacency=adjacency,
                degrees=degrees,
                start_nodes=second_round_starts,
            )
            best_result = min([best_result, *second_round_results], key=lambda result: (result[2], result[3]))

    return best_result[0], best_result[1], best_result[2]


def renumber_nodes_multilevel_rcm(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    leaf_size: int = 75,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    if not nodes:
        return [], [], 0
    if not triangles:
        return list(nodes), [], 0
    if leaf_size < 1:
        raise ValueError("leaf_size must be >= 1")

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    degrees = [len(neighbors) for neighbors in adjacency]
    all_nodes = list(range(len(nodes)))
    order = _build_multilevel_rcm_order(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        adjacency=adjacency,
        degrees=degrees,
        subset=all_nodes,
        leaf_size=leaf_size,
    )

    if len(order) < len(nodes):
        seen = set(order)
        missing = [node_id for node_id in all_nodes if node_id not in seen]
        missing.sort(key=lambda node_id: (degrees[node_id], node_id))
        order.extend(missing)

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(order)}
    new_nodes = [nodes[old_id] for old_id in order]
    remapped_triangles_zero_based = [
        tuple(old_to_new[node_id] for node_id in triangle)
        for triangle in normalized_triangles
    ]
    bandwidth = compute_max_difference(
        _build_linear_triangles_from_indexed_mesh(new_nodes, _restore_triangle_indices(remapped_triangles_zero_based, 1), index_base=1)
    )
    new_triangles = _restore_triangle_indices(remapped_triangles_zero_based, index_base)
    return new_nodes, new_triangles, bandwidth


def renumber_nodes_gps(
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
    start_node, end_node = _find_gps_pseudo_diameter(adjacency, degrees)
    gps_order = _build_gps_order(adjacency, degrees, start_node=start_node, end_node=end_node)
    if len(gps_order) < len(nodes):
        seen = set(gps_order)
        remaining = [node_id for node_id in range(len(nodes)) if node_id not in seen]
        remaining.sort(key=lambda node_id: (degrees[node_id], node_id))
        gps_order.extend(remaining)

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(gps_order)}
    new_nodes = [nodes[old_id] for old_id in gps_order]
    remapped_triangles_zero_based = [
        tuple(old_to_new[node_id] for node_id in triangle)
        for triangle in normalized_triangles
    ]
    bandwidth = _compute_bandwidth(remapped_triangles_zero_based)
    new_triangles = _restore_triangle_indices(remapped_triangles_zero_based, index_base)
    return new_nodes, new_triangles, bandwidth


def refine_numbering_local_search(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    node_numbering: list[int] | None = None,
    max_iterations: int = 64,
    no_improvement_limit: int = 16,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    if not nodes:
        return [], [], 0
    if not triangles:
        return list(nodes), [], 0
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if no_improvement_limit < 1:
        raise ValueError("no_improvement_limit must be >= 1")

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    current_nodes = list(nodes)
    if node_numbering is None:
        working_triangles = [tuple(node_id - index_base for node_id in triangle) for triangle in normalized_triangles]
    else:
        current_labels = list(node_numbering)
        if len(current_labels) != len(nodes):
            raise ValueError("node_numbering must contain exactly one label per node")
        label_to_position = {label: position for position, label in enumerate(current_labels)}
        if len(label_to_position) != len(current_labels):
            raise ValueError("node_numbering must contain unique labels")
        working_triangles = [
            tuple(label_to_position[node_id] for node_id in triangle)
            for triangle in normalized_triangles
        ]
    current_beta = _compute_bandwidth(working_triangles)
    adjacency = _build_node_adjacency(len(nodes), working_triangles)

    attempts = 0
    stagnant_attempts = 0

    while attempts < max_iterations and stagnant_attempts < no_improvement_limit:
        worst_triangles = [triangle for triangle in working_triangles if _triangle_difference(triangle) == current_beta]
        if not worst_triangles:
            break

        improved = False
        for triangle in worst_triangles:
            candidate_pairs = _local_search_swap_candidates(triangle, adjacency)
            for source_node, target_node in candidate_pairs:
                if source_node == target_node:
                    continue
                attempts += 1
                swapped_triangles = _swap_triangle_labels(working_triangles, source_node, target_node)
                new_beta = _compute_bandwidth(swapped_triangles)
                if new_beta >= current_beta:
                    stagnant_attempts += 1
                    if attempts >= max_iterations or stagnant_attempts >= no_improvement_limit:
                        break
                    continue

                current_beta = new_beta
                working_triangles = swapped_triangles
                current_nodes[source_node], current_nodes[target_node] = current_nodes[target_node], current_nodes[source_node]
                adjacency = _build_node_adjacency(len(nodes), working_triangles)
                stagnant_attempts = 0
                improved = True
                break

            if improved or attempts >= max_iterations or stagnant_attempts >= no_improvement_limit:
                break

        if not improved:
            break

    refined_triangles = [tuple(node_id + index_base for node_id in triangle) for triangle in working_triangles]
    return current_nodes, refined_triangles, current_beta


def renumber_nodes_sloan(
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
    start_node, end_node = _find_pseudo_peripheral_pair(adjacency, degrees)
    return _renumber_nodes_sloan_from_pair(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        index_base=index_base,
        adjacency=adjacency,
        degrees=degrees,
        start_node=start_node,
        end_node=end_node,
    )


def renumber_nodes_sloan_multistart(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    max_candidates: int = 5,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    if not nodes:
        return [], [], 0
    if not triangles:
        return list(nodes), [], 0
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    normalized_triangles, index_base = _normalize_triangle_indices(triangles, len(nodes))
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    degrees = [len(neighbors) for neighbors in adjacency]
    seeds = _select_sloan_seed_nodes(adjacency, degrees, max_candidates=max_candidates)

    candidate_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for seed in seeds:
        pair = _find_pseudo_peripheral_pair(adjacency, degrees, start_node=seed)
        if pair in seen_pairs:
            continue
        candidate_pairs.append(pair)
        seen_pairs.add(pair)
        if len(candidate_pairs) >= max_candidates:
            break

    if not candidate_pairs:
        candidate_pairs.append(_find_pseudo_peripheral_pair(adjacency, degrees))

    best_result: tuple[list[tuple[float, float]], list[tuple[int, int, int]], int, tuple[int, int]] | None = None
    for start_node, end_node in candidate_pairs:
        renumbered_nodes, renumbered_triangles, _ = _renumber_nodes_sloan_from_pair(
            nodes=nodes,
            normalized_triangles=normalized_triangles,
            index_base=index_base,
            adjacency=adjacency,
            degrees=degrees,
            start_node=start_node,
            end_node=end_node,
        )
        beta = _compute_bandwidth(renumbered_triangles)
        candidate_result = (renumbered_nodes, renumbered_triangles, beta, (start_node, end_node))
        if best_result is None or (beta, start_node, end_node) < (best_result[2], best_result[3][0], best_result[3][1]):
            best_result = candidate_result

    assert best_result is not None
    return best_result[0], best_result[1], best_result[2]


def laplacian_smooth(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    boundary_nodes: set[int],
    interface_nodes: set[int] | None = None,
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
    normalized_interface = _normalize_boundary_indices(interface_nodes or set(), len(nodes), index_base)
    fixed_nodes = normalized_boundary | normalized_interface
    adjacency = _build_node_adjacency(len(nodes), normalized_triangles)
    incident_triangles = _build_incident_triangles(len(nodes), normalized_triangles)

    positions = [Point(float(x), float(y)) for x, y in nodes]
    internal_nodes = [node_id for node_id in range(len(nodes)) if node_id not in fixed_nodes]

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


def _find_gps_pseudo_diameter(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int | None = None,
) -> tuple[int, int]:
    if not adjacency:
        return 0, 0

    current = _select_min_degree_start(degrees) if start_node is None else start_node
    best_start = current
    best_end = current

    while True:
        current_dist = _distances_from_node(adjacency, current)
        current_depth = _level_depth(current_dist)
        last_level_nodes = [node_id for node_id, level in enumerate(current_dist) if level == current_depth]
        if not last_level_nodes:
            break

        ordered_last_level = sorted(last_level_nodes, key=lambda node_id: (degrees[node_id], node_id))
        improved = False
        best_candidate = ordered_last_level[0]
        best_candidate_depth = -1
        best_candidate_width = float("inf")

        for candidate in ordered_last_level:
            candidate_dist = _distances_from_node(adjacency, candidate)
            candidate_depth = _level_depth(candidate_dist)
            candidate_width = _level_width(candidate_dist, candidate_depth)
            if candidate_depth > current_depth:
                current = candidate
                improved = True
                break
            if candidate_depth < best_candidate_depth:
                continue
            candidate_rank = (candidate_width, degrees[candidate], candidate)
            best_rank = (best_candidate_width, degrees[best_candidate], best_candidate)
            if candidate_rank < best_rank:
                best_candidate = candidate
                best_candidate_depth = candidate_depth
                best_candidate_width = candidate_width

        if improved:
            best_start = current
            continue

        best_start = current
        best_end = best_candidate
        break

    return best_start, best_end


def _build_gps_order(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int,
    end_node: int,
) -> list[int]:
    if degrees[end_node] < degrees[start_node]:
        start_node, end_node = end_node, start_node

    dist_from_start = _distances_from_node(adjacency, start_node)
    dist_from_end = _distances_from_node(adjacency, end_node)
    if any(distance < 0 for distance in dist_from_start) or any(distance < 0 for distance in dist_from_end):
        return list(reversed(_build_bfs_order(adjacency, degrees, start_node=start_node)))

    level_count = max(_level_depth(dist_from_start), _level_depth(dist_from_end)) + 1
    level_pairs = [
        (dist_from_start[node_id] + 1, level_count - dist_from_end[node_id])
        for node_id in range(len(adjacency))
    ]

    assigned_levels: list[list[int]] = [[] for _ in range(level_count + 1)]
    assigned = [False] * len(adjacency)

    for node_id, (first_level, second_level) in enumerate(level_pairs):
        if first_level == second_level:
            assigned_levels[first_level].append(node_id)
            assigned[node_id] = True

    remaining_nodes = [node_id for node_id in range(len(adjacency)) if not assigned[node_id]]
    components = _build_component_list(adjacency, remaining_nodes)
    component_counts = [len(level_nodes) for level_nodes in assigned_levels]

    for component in components:
        first_counts = list(component_counts)
        second_counts = list(component_counts)
        for node_id in component:
            first_level, second_level = level_pairs[node_id]
            first_counts[first_level] += 1
            second_counts[second_level] += 1

        first_metric = (max(first_counts), sum(level * count for level, count in enumerate(first_counts)), len(component))
        second_metric = (max(second_counts), sum(level * count for level, count in enumerate(second_counts)), len(component))
        use_first = first_metric <= second_metric

        for node_id in component:
            level = level_pairs[node_id][0 if use_first else 1]
            assigned_levels[level].append(node_id)
            assigned[node_id] = True

        component_counts = first_counts if use_first else second_counts

    if not all(assigned):
        fallback_nodes = [node_id for node_id in range(len(adjacency)) if not assigned[node_id]]
        fallback_nodes.sort(key=lambda node_id: (degrees[node_id], node_id))
        for node_id in fallback_nodes:
            assigned_levels[1].append(node_id)

    order: list[int] = []
    for level in range(1, level_count + 1):
        assigned_levels[level].sort(key=lambda node_id: (degrees[node_id], node_id))
        order.extend(assigned_levels[level])

    return order


def _build_component_list(adjacency: list[set[int]], allowed_nodes: list[int]) -> list[list[int]]:
    allowed = set(allowed_nodes)
    components: list[list[int]] = []

    while allowed:
        start = min(allowed)
        queue: deque[int] = deque([start])
        allowed.remove(start)
        component: list[int] = []

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in allowed:
                    continue
                allowed.remove(neighbor)
                queue.append(neighbor)

        component.sort(key=lambda node_id: (len(adjacency[node_id]), node_id))
        components.append(component)

    components.sort(key=lambda component: (-len(component), component[0] if component else -1))
    return components


def _build_multilevel_rcm_order(
    nodes: list[tuple[float, float]],
    normalized_triangles: list[TriangleIndices],
    adjacency: list[set[int]],
    degrees: list[int],
    subset: list[int],
    leaf_size: int,
) -> list[int]:
    if not subset:
        return []
    if len(subset) <= leaf_size:
        return _leaf_rcm_order(nodes, normalized_triangles, degrees, subset)

    left_subset, right_subset = _bisect_multilevel_subset(subset, adjacency, degrees)
    if not left_subset or not right_subset:
        return _leaf_rcm_order(nodes, normalized_triangles, degrees, subset)

    left_order = _build_multilevel_rcm_order(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        adjacency=adjacency,
        degrees=degrees,
        subset=left_subset,
        leaf_size=leaf_size,
    )
    right_order = _build_multilevel_rcm_order(
        nodes=nodes,
        normalized_triangles=normalized_triangles,
        adjacency=adjacency,
        degrees=degrees,
        subset=right_subset,
        leaf_size=leaf_size,
    )
    return [*left_order, *right_order]


def _leaf_rcm_order(
    nodes: list[tuple[float, float]],
    normalized_triangles: list[TriangleIndices],
    degrees: list[int],
    subset: list[int],
) -> list[int]:
    subset_set = set(subset)
    local_index = {global_id: local_id for local_id, global_id in enumerate(subset)}
    local_nodes = [nodes[global_id] for global_id in subset]
    local_triangles = [
        tuple(local_index[node_id] for node_id in triangle)
        for triangle in normalized_triangles
        if all(node_id in subset_set for node_id in triangle)
    ]
    local_adjacency = _build_node_adjacency(len(local_nodes), local_triangles)
    local_degrees = [len(neighbors) for neighbors in local_adjacency]
    if not local_nodes:
        return []
    start_node = _select_min_degree_start(local_degrees)
    renumbered_nodes, _, _ = _renumber_nodes_rcm_from_start(
        nodes=local_nodes,
        normalized_triangles=local_triangles,
        index_base=0,
        adjacency=local_adjacency,
        degrees=local_degrees,
        start_node=start_node,
    )
    if not renumbered_nodes:
        return subset

    local_index_by_node = {node: index for index, node in enumerate(local_nodes)}
    ordered_local_ids = [local_index_by_node[local_node] for local_node in renumbered_nodes]
    return [subset[local_id] for local_id in ordered_local_ids]


def _bisect_multilevel_subset(
    subset: list[int],
    adjacency: list[set[int]],
    degrees: list[int],
) -> tuple[list[int], list[int]]:
    subset_set = set(subset)
    root = _find_multilevel_root(subset, adjacency, degrees)
    distances = _restricted_distances_from_node(adjacency, subset_set, root)

    ordered = sorted(
        subset,
        key=lambda node_id: (
            distances[node_id] if distances[node_id] >= 0 else len(subset) + 1,
            degrees[node_id],
            node_id,
        ),
    )
    if len(ordered) < 2:
        return ordered, []

    level_counts: list[tuple[int, int]] = []
    current_level = None
    count = 0
    for node_id in ordered:
        level = distances[node_id] if distances[node_id] >= 0 else len(subset) + 1
        if current_level is None:
            current_level = level
            count = 1
            continue
        if level == current_level:
            count += 1
            continue
        level_counts.append((current_level, count))
        current_level = level
        count = 1
    if current_level is not None:
        level_counts.append((current_level, count))

    total = len(subset)
    threshold = total / 2.0
    cumulative = 0
    cutoff_level = level_counts[-1][0]
    for level, count in level_counts:
        cumulative += count
        cutoff_level = level
        if cumulative >= threshold:
            break

    left = [node_id for node_id in ordered if (distances[node_id] if distances[node_id] >= 0 else len(subset) + 1) <= cutoff_level]
    left_set = set(left)
    right = [node_id for node_id in ordered if node_id not in left_set]
    if not left or not right:
        midpoint = max(1, total // 2)
        left = ordered[:midpoint]
        right = ordered[midpoint:]
    return left, right


def _find_multilevel_root(
    subset: list[int],
    adjacency: list[set[int]],
    degrees: list[int],
) -> int:
    if not subset:
        return 0

    current = min(subset, key=lambda node_id: (degrees[node_id], node_id))
    best_root = current
    best_depth = -1
    seen_roots: set[int] = set()

    while current not in seen_roots:
        seen_roots.add(current)
        distances = _restricted_distances_from_node(adjacency, set(subset), current)
        depth = _level_depth(distances)
        if depth > best_depth:
            best_depth = depth
            best_root = current

        last_level_nodes = [
            node_id
            for node_id in subset
            if distances[node_id] == depth
        ]
        if not last_level_nodes:
            break

        next_current = min(last_level_nodes, key=lambda node_id: (degrees[node_id], node_id))
        if next_current == current:
            break
        current = next_current

    return best_root


def _restricted_distances_from_node(
    adjacency: list[set[int]],
    allowed_nodes: set[int],
    start_node: int,
) -> list[int]:
    distances = [-1] * len(adjacency)
    if start_node not in allowed_nodes:
        return distances

    queue: deque[int] = deque([start_node])
    distances[start_node] = 0

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in allowed_nodes or distances[neighbor] != -1:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _select_min_degree_start(degrees: list[int]) -> int:
    return min(range(len(degrees)), key=lambda node_id: (degrees[node_id], node_id))


def _select_sloan_seed_nodes(adjacency: list[set[int]], degrees: list[int], max_candidates: int) -> list[int]:
    ordered = sorted(range(len(degrees)), key=lambda node_id: (degrees[node_id], node_id))
    if not ordered:
        return []

    min_degree = degrees[ordered[0]]
    low_degree_nodes = [node_id for node_id in ordered if degrees[node_id] == min_degree]

    probe_budget = min(2, max_candidates - 1)
    base_budget = max_candidates - probe_budget if probe_budget > 0 else max_candidates
    seeds = low_degree_nodes[:base_budget]
    seen: set[int] = set(seeds)

    probe_starts: list[int] = []
    for node_id in (max(0, len(degrees) // 20), max(0, len(degrees) // 4)):
        if len(probe_starts) >= probe_budget:
            break
        if not 0 <= node_id < len(degrees):
            continue
        if degrees[node_id] == min_degree or node_id in seen or node_id in probe_starts:
            continue
        probe_starts.append(node_id)

    if len(probe_starts) < probe_budget:
        median_start = ordered[len(ordered) // 2]
        if degrees[median_start] != min_degree and median_start not in seen and median_start not in probe_starts:
            probe_starts.append(median_start)

    for probe_start in probe_starts:
        _, end_node = _find_pseudo_peripheral_pair(adjacency, degrees, start_node=probe_start)
        if end_node in seen:
            continue
        seeds.append(end_node)
        seen.add(end_node)
        if len(seeds) >= max_candidates:
            return seeds[:max_candidates]

    if len(seeds) < max_candidates:
        for node_id in low_degree_nodes[base_budget:]:
            if node_id in seen:
                continue
            seeds.append(node_id)
            seen.add(node_id)
            if len(seeds) >= max_candidates:
                break
    return seeds[:max_candidates]


def _normalize_rcm_candidates(
    candidates: list[int] | None,
    degrees: list[int],
    max_candidates: int,
) -> list[int]:
    if candidates is None:
        min_degree = min(degrees)
        min_degree_nodes = [node_id for node_id, degree in enumerate(degrees) if degree == min_degree]
        if len(min_degree_nodes) <= max_candidates:
            return min_degree_nodes
        ordered_nodes = sorted(range(len(degrees)), key=lambda node_id: (degrees[node_id], node_id))
        return ordered_nodes[:max_candidates]

    normalized_candidates: list[int] = []
    seen: set[int] = set()
    for node_id in sorted(candidates):
        if not isinstance(node_id, int):
            raise ValueError("candidate start nodes must be integers")
        if not 0 <= node_id < len(degrees):
            raise ValueError("candidate start nodes are out of bounds for the provided graph")
        if node_id in seen:
            continue
        normalized_candidates.append(node_id)
        seen.add(node_id)
    if not normalized_candidates:
        raise ValueError("candidates must contain at least one valid start node")
    return normalized_candidates


def _expand_rcm_candidate_pool(
    adjacency: list[set[int]],
    degrees: list[int],
    candidate_nodes: list[int],
    max_candidates: int,
    probe_stride: int,
) -> list[int]:
    expanded = list(candidate_nodes)
    seen = set(expanded)

    for node_id in _select_rcm_probe_starts(len(degrees), probe_stride):
        if node_id in seen:
            continue
        expanded.append(node_id)
        seen.add(node_id)
        if len(expanded) >= max_candidates:
            return expanded[:max_candidates]

    ecc_candidates = _collect_rcm_eccentricity_candidates(adjacency, degrees, probe_budget=3)
    for node_id in ecc_candidates:
        if node_id in seen:
            continue
        expanded.append(node_id)
        seen.add(node_id)
        if len(expanded) >= max_candidates:
            return expanded[:max_candidates]

    return expanded[:max_candidates]


def _expand_rcm_refinement_starts(
    adjacency: list[set[int]],
    degrees: list[int],
    first_round_results: list[tuple[list[tuple[float, float]], list[tuple[int, int, int]], int, int]],
    max_candidates: int,
) -> list[int]:
    top_results = sorted(first_round_results, key=lambda result: (result[2], result[3]))[:3]
    seen: set[int] = set()
    starts: list[int] = []
    for _, _, _, start_node in top_results:
        neighbors = sorted(adjacency[start_node], key=lambda node_id: (degrees[node_id], node_id))
        for node_id in neighbors:
            if node_id in seen:
                continue
            seen.add(node_id)
            starts.append(node_id)
            if len(starts) >= max_candidates:
                return starts[:max_candidates]
    return starts[:max_candidates]


def _evaluate_rcm_candidates(
    nodes: list[tuple[float, float]],
    normalized_triangles: list[TriangleIndices],
    index_base: int,
    adjacency: list[set[int]],
    degrees: list[int],
    start_nodes: list[int],
) -> list[tuple[list[tuple[float, float]], list[tuple[int, int, int]], int, int]]:
    results: list[tuple[list[tuple[float, float]], list[tuple[int, int, int]], int, int]] = []
    for start_node in start_nodes:
        renumbered_nodes, renumbered_triangles, _ = _renumber_nodes_rcm_from_start(
            nodes=nodes,
            normalized_triangles=normalized_triangles,
            index_base=index_base,
            adjacency=adjacency,
            degrees=degrees,
            start_node=start_node,
        )
        beta = compute_max_difference(
            _build_linear_triangles_from_indexed_mesh(renumbered_nodes, renumbered_triangles, index_base=index_base)
        )
        results.append((renumbered_nodes, renumbered_triangles, beta, start_node))
    return results


def _select_rcm_probe_starts(node_count: int, probe_stride: int) -> list[int]:
    if node_count <= 0 or probe_stride <= 0:
        return []

    starts: list[int] = []
    anchors = [
        max(0, node_count // 14),
        max(0, node_count // 7),
        max(0, node_count // 3),
        max(0, (2 * node_count) // 3),
    ]
    for anchor in anchors:
        start = min(node_count - 1, max(0, anchor))
        if start not in starts:
            starts.append(start)

    stride_starts = list(range(max(1, probe_stride // 2), node_count, probe_stride))
    for start in stride_starts:
        if start not in starts:
            starts.append(start)
        if len(starts) >= 6:
            break
    return starts


def _collect_rcm_eccentricity_candidates(
    adjacency: list[set[int]],
    degrees: list[int],
    probe_budget: int = 3,
) -> list[int]:
    if not adjacency or probe_budget < 1:
        return []

    ordered = sorted(range(len(degrees)), key=lambda node_id: (degrees[node_id], node_id))
    baseline = {ordered[0]}
    probe_starts: list[int] = []

    preferred = [
        ordered[len(ordered) // 6],
        ordered[len(ordered) // 2],
        ordered[(5 * len(ordered)) // 6],
    ]
    for node_id in preferred:
        if node_id in baseline or node_id in probe_starts:
            continue
        probe_starts.append(node_id)
        if len(probe_starts) >= probe_budget:
            break

    if len(probe_starts) < probe_budget:
        for node_id in ordered:
            if node_id in baseline or node_id in probe_starts:
                continue
            probe_starts.append(node_id)
            if len(probe_starts) >= probe_budget:
                break

    scored: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for probe_start in probe_starts:
        end_node, ecc = _find_rcm_pseudo_peripheral_pair(adjacency, degrees, start_node=probe_start)
        if end_node in seen:
            continue
        scored.append((ecc, degrees[end_node], end_node))
        seen.add(end_node)

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [node_id for _, _, node_id in scored]


def _find_rcm_pseudo_peripheral_pair(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int,
) -> tuple[int, int]:
    current = start_node
    best_end = start_node
    best_ecc = _level_depth(_distances_from_node(adjacency, current))

    while True:
        dist = _distances_from_node(adjacency, current)
        depth = _level_depth(dist)
        last_level_nodes = [node_id for node_id, level in enumerate(dist) if level == depth]
        if not last_level_nodes:
            return best_end, best_ecc

        ordered_last_level = sorted(last_level_nodes, key=lambda node_id: (degrees[node_id], node_id))
        candidate_end = ordered_last_level[0]
        candidate_ecc = -1
        candidate_rank = (degrees[candidate_end], candidate_end)

        for node_id in ordered_last_level:
            node_dist = _distances_from_node(adjacency, node_id)
            node_ecc = _level_depth(node_dist)
            node_rank = (degrees[node_id], node_id)
            if node_ecc > candidate_ecc or (node_ecc == candidate_ecc and node_rank < candidate_rank):
                candidate_end = node_id
                candidate_ecc = node_ecc
                candidate_rank = node_rank

        if candidate_ecc <= best_ecc:
            return best_end, best_ecc

        best_end = candidate_end
        best_ecc = candidate_ecc
        if candidate_end == current:
            return best_end, best_ecc
        current = candidate_end


def _renumber_nodes_rcm_from_start(
    nodes: list[tuple[float, float]],
    normalized_triangles: list[TriangleIndices],
    index_base: int,
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    bfs_order = _build_bfs_order(adjacency, degrees, start_node=start_node)
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


def _renumber_nodes_sloan_from_pair(
    nodes: list[tuple[float, float]],
    normalized_triangles: list[TriangleIndices],
    index_base: int,
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int,
    end_node: int,
    weights: tuple[int, int] = (2, 1),
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
    order = _build_sloan_order(adjacency, degrees, start_node=start_node, end_node=end_node, weights=weights)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(order)}

    new_nodes = [nodes[old_id] for old_id in order]
    remapped_triangles_zero_based = [
        tuple(old_to_new[node_id] for node_id in triangle)
        for triangle in normalized_triangles
    ]
    bandwidth = _compute_bandwidth(remapped_triangles_zero_based)
    new_triangles = _restore_triangle_indices(remapped_triangles_zero_based, index_base)
    return new_nodes, new_triangles, bandwidth


def _build_sloan_order(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int,
    end_node: int,
    weights: tuple[int, int] = (2, 1),
) -> list[int]:
    w_distance, w_degree = weights
    dist_to_end = _distances_from_node(adjacency, end_node)
    n = len(adjacency)

    WHITE, GREEN, RED, BLACK = 0, 1, 2, 3
    color = [WHITE] * n
    priority = [w_distance * dist_to_end[node_id] - w_degree * (degrees[node_id] + 1) for node_id in range(n)]
    version = [0] * n
    heap: list[tuple[int, int, int]] = []
    order: list[int] = []

    def push(node_id: int) -> None:
        heapq.heappush(heap, (-priority[node_id], node_id, version[node_id]))

    color[start_node] = GREEN
    push(start_node)

    while heap:
        _, node_id, seen_version = heapq.heappop(heap)
        if seen_version != version[node_id] or color[node_id] == WHITE:
            continue

        if color[node_id] == GREEN:
            for neighbor in adjacency[node_id]:
                priority[neighbor] += w_degree
                version[neighbor] += 1
                if color[neighbor] == WHITE:
                    color[neighbor] = GREEN
                    push(neighbor)
                elif color[neighbor] != BLACK:
                    push(neighbor)

        order.append(node_id)
        color[node_id] = BLACK

        for neighbor in adjacency[node_id]:
            if color[neighbor] == GREEN:
                color[neighbor] = RED
                priority[neighbor] += w_degree
                version[neighbor] += 1
                push(neighbor)

                for second_neighbor in adjacency[neighbor]:
                    if color[second_neighbor] == BLACK:
                        continue
                    priority[second_neighbor] += w_degree
                    version[second_neighbor] += 1
                    if color[second_neighbor] == WHITE:
                        color[second_neighbor] = GREEN
                    push(second_neighbor)

    if len(order) < n:
        remaining = [node_id for node_id in range(n) if color[node_id] != BLACK]
        remaining.sort(key=lambda node_id: (-priority[node_id], node_id))
        for node_id in remaining:
            if color[node_id] == BLACK:
                continue
            order.append(node_id)
            color[node_id] = BLACK

    return order


def _find_pseudo_peripheral_pair(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int | None = None,
) -> tuple[int, int]:
    if not adjacency:
        return 0, 0

    s = _select_min_degree_start(degrees) if start_node is None else start_node
    best_start = s
    best_end = s
    best_ecc = -1

    while True:
        dist = _distances_from_node(adjacency, s)
        h_s = _level_depth(dist)
        last_level_nodes = [node_id for node_id, level in enumerate(dist) if level == h_s]
        if not last_level_nodes:
            break

        candidate_end = last_level_nodes[0]
        candidate_ecc = -1
        candidate_rank = (degrees[candidate_end], candidate_end)

        for node_id in sorted(last_level_nodes, key=lambda node_id: (degrees[node_id], node_id)):
            dist_i = _distances_from_node(adjacency, node_id)
            h_i = _level_depth(dist_i)
            rank = (degrees[node_id], node_id)
            if h_i > candidate_ecc or (h_i == candidate_ecc and rank < candidate_rank):
                candidate_end = node_id
                candidate_ecc = h_i
                candidate_rank = rank

        if candidate_ecc <= best_ecc:
            break

        best_start = s
        best_end = candidate_end
        best_ecc = candidate_ecc
        if candidate_end == s:
            break
        s = candidate_end

    return best_start, best_end


def _distances_from_node(adjacency: list[set[int]], start_node: int) -> list[int]:
    dist = [-1] * len(adjacency)
    queue: deque[int] = deque([start_node])
    dist[start_node] = 0

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if dist[neighbor] != -1:
                continue
            dist[neighbor] = dist[current] + 1
            queue.append(neighbor)
    return dist


def _level_depth(distances: list[int]) -> int:
    return max((distance for distance in distances if distance >= 0), default=0)


def _level_width(distances: list[int], depth: int) -> int:
    counts = [0] * (depth + 1 if depth >= 0 else 1)
    for distance in distances:
        if distance >= 0:
            counts[distance] += 1
    return max(counts, default=0)


def _build_bfs_order(
    adjacency: list[set[int]],
    degrees: list[int],
    start_node: int | None = None,
) -> list[int]:
    bfs_order: list[int] = []
    visited = [False] * len(adjacency)

    def traverse(start: int) -> None:
        if visited[start]:
            return
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

    if start_node is not None:
        traverse(start_node)

    while len(bfs_order) < len(adjacency):
        remaining = [node_id for node_id in range(len(adjacency)) if not visited[node_id]]
        start = min(remaining, key=lambda node_id: (degrees[node_id], node_id))
        traverse(start)

    return bfs_order


def _build_linear_triangles_from_indexed_mesh(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    index_base: int,
) -> list[LinearTriangle]:
    point_by_id = [Point(x, y) for x, y in nodes]
    linear_triangles: list[LinearTriangle] = []
    for triangle_number, triangle in enumerate(triangles, start=1):
        coordinates = tuple(point_by_id[node_id - index_base] for node_id in triangle)
        linear_triangles.append(
            LinearTriangle(
                triangle_number=triangle_number,
                node_coordinates=coordinates,  # type: ignore[arg-type]
                node_numbers=tuple(node_id if index_base == 1 else node_id + 1 for node_id in triangle),
            )
        )
    return linear_triangles


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


def _local_search_swap_candidates(
    triangle: TriangleIndices,
    adjacency: list[set[int]],
) -> list[tuple[int, int]]:
    ordered_nodes = sorted(triangle)
    low_node, mid_node, high_node = ordered_nodes

    candidates: list[tuple[int, int]] = []
    for target_node in sorted(node_id for node_id in adjacency[high_node] if node_id < high_node):
        candidates.append((high_node, target_node))
    for target_node in sorted(node_id for node_id in adjacency[low_node] if node_id > low_node):
        candidates.append((low_node, target_node))
    for target_node in sorted(adjacency[mid_node]):
        if target_node not in (low_node, mid_node, high_node):
            candidates.append((mid_node, target_node))

    seen: set[tuple[int, int]] = set()
    unique_candidates: list[tuple[int, int]] = []
    for source_node, target_node in candidates:
        pair = (source_node, target_node)
        if pair in seen:
            continue
        seen.add(pair)
        unique_candidates.append(pair)
    return unique_candidates


def _swap_triangle_labels(
    triangles: list[TriangleIndices],
    first_label: int,
    second_label: int,
) -> list[TriangleIndices]:
    remap = {first_label: second_label, second_label: first_label}
    return [tuple(remap.get(node_id, node_id) for node_id in triangle) for triangle in triangles]


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
