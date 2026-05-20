from __future__ import annotations

from collections import deque

Mask = list[list[int]]
Pixel = tuple[int, int]
Contour = list[Pixel]

_NEIGHBORS_8 = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def extract_stroke_centerlines(mask: Mask) -> list[Contour]:
    _validate_mask(mask)
    skeleton = _zhang_suen_thinning(mask)
    components = _connected_components(skeleton)
    contours: list[Contour] = []
    for component in components:
        contour = _trace_component(component)
        if len(contour) >= 2:
            contours.append(contour)
    return contours


def _validate_mask(mask: Mask) -> None:
    if not mask or not mask[0]:
        raise ValueError("mask must not be empty")
    width = len(mask[0])
    for row in mask:
        if len(row) != width:
            raise ValueError("mask must be rectangular")


def _zhang_suen_thinning(mask: Mask) -> Mask:
    working = [row[:] for row in mask]
    height = len(working)
    width = len(working[0])

    changed = True
    while changed:
        changed = False
        to_remove: list[Pixel] = []
        for step in (0, 1):
            to_remove.clear()
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if working[y][x] != 1:
                        continue
                    neighbors = _neighbor_values(working, x, y)
                    black_neighbors = sum(neighbors)
                    transitions = _white_to_black_transitions(neighbors)
                    if black_neighbors < 2 or black_neighbors > 6:
                        continue
                    if transitions != 1:
                        continue
                    p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((x, y))
            if to_remove:
                changed = True
                for x, y in to_remove:
                    working[y][x] = 0
    return working


def _neighbor_values(mask: Mask, x: int, y: int) -> list[int]:
    return [
        mask[y - 1][x],
        mask[y - 1][x + 1],
        mask[y][x + 1],
        mask[y + 1][x + 1],
        mask[y + 1][x],
        mask[y + 1][x - 1],
        mask[y][x - 1],
        mask[y - 1][x - 1],
    ]


def _white_to_black_transitions(neighbors: list[int]) -> int:
    extended = neighbors + [neighbors[0]]
    transitions = 0
    for idx in range(len(neighbors)):
        if extended[idx] == 0 and extended[idx + 1] == 1:
            transitions += 1
    return transitions


def _connected_components(mask: Mask) -> list[set[Pixel]]:
    height = len(mask)
    width = len(mask[0])
    visited: set[Pixel] = set()
    components: list[set[Pixel]] = []

    for y in range(height):
        for x in range(width):
            if mask[y][x] != 1 or (x, y) in visited:
                continue
            component: set[Pixel] = set()
            queue: deque[Pixel] = deque([(x, y)])
            visited.add((x, y))
            while queue:
                current = queue.popleft()
                component.add(current)
                for neighbor in _neighbors_in_component(current, mask):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            components.append(component)
    return components


def _neighbors_in_component(pixel: Pixel, mask: Mask) -> list[Pixel]:
    x, y = pixel
    height = len(mask)
    width = len(mask[0])
    neighbors: list[Pixel] = []
    for dx, dy in _NEIGHBORS_8:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] == 1:
            neighbors.append((nx, ny))
    return neighbors


def _trace_component(component: set[Pixel]) -> Contour:
    adjacency = {pixel: _component_neighbors(pixel, component) for pixel in component}
    endpoints = [pixel for pixel, neighbors in adjacency.items() if len(neighbors) == 1]

    if endpoints:
        start = min(endpoints, key=lambda item: (item[1], item[0]))
    else:
        start = min(component, key=lambda item: (item[1], item[0]))

    contour: Contour = [start]
    visited_edges: set[frozenset[Pixel]] = set()
    previous: Pixel | None = None
    current = start

    while True:
        candidates = []
        for neighbor in adjacency[current]:
            edge = frozenset((current, neighbor))
            if edge in visited_edges:
                continue
            candidates.append(neighbor)
        if not candidates:
            break

        next_pixel = _pick_next_pixel(previous, current, candidates)
        visited_edges.add(frozenset((current, next_pixel)))
        contour.append(next_pixel)
        previous, current = current, next_pixel
        if current == start:
            break

    return contour


def _component_neighbors(pixel: Pixel, component: set[Pixel]) -> list[Pixel]:
    x, y = pixel
    neighbors: list[Pixel] = []
    for dx, dy in _NEIGHBORS_8:
        neighbor = (x + dx, y + dy)
        if neighbor in component:
            neighbors.append(neighbor)
    return neighbors


def _pick_next_pixel(previous: Pixel | None, current: Pixel, candidates: list[Pixel]) -> Pixel:
    if previous is None or len(candidates) == 1:
        return min(candidates, key=lambda item: (item[1], item[0]))

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]

    def score(candidate: Pixel) -> tuple[float, int, int]:
        ndx = candidate[0] - current[0]
        ndy = candidate[1] - current[1]
        dot = dx * ndx + dy * ndy
        return (-dot, candidate[1], candidate[0])

    return min(candidates, key=score)
