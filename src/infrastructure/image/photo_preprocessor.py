from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.domain.entities.point import Point

Mask = list[list[int]]
Pixel = tuple[int, int]
Contour = list[Pixel]
Edge = tuple[Pixel, Pixel]
_EPSILON = 1e-12


@dataclass(slots=True)
class ImagePreprocessingResult:
    mask: Mask
    external_contour: Contour
    simplified_contour: Contour
    boundary: list[Point]


def load_binary_mask(image_path: str | Path, threshold: int = 127, invert: bool = False) -> Mask:
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255]")

    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        width, height = grayscale.size
        pixels = grayscale.load()

        mask: Mask = []
        for y in range(height):
            row: list[int] = []
            for x in range(width):
                value = 1 if pixels[x, y] > threshold else 0
                if invert:
                    value = 1 - value
                row.append(value)
            mask.append(row)
    return mask


def find_external_contour(mask: Mask) -> Contour:
    contours = _extract_contours(mask)
    if not contours:
        raise ValueError("No foreground contour found in mask.")
    return max(contours, key=lambda contour: abs(_polygon_area(contour)))


def simplify_contour(contour: Contour, epsilon: float) -> Contour:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(contour) < 4 or epsilon == 0:
        return contour.copy()

    closed = contour + [contour[0]]
    simplified = _rdp(closed, epsilon)

    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]

    if len(simplified) < 3:
        return contour.copy()

    return simplified


def contour_to_boundary(contour: Contour) -> list[Point]:
    return [Point(float(x), float(y)) for x, y in contour]


def preprocess_photo_to_boundary(
    image_path: str | Path,
    threshold: int = 127,
    epsilon: float = 1.5,
    invert: bool = False,
) -> ImagePreprocessingResult:
    mask = load_binary_mask(image_path=image_path, threshold=threshold, invert=invert)
    external_contour = find_external_contour(mask)
    simplified_contour = simplify_contour(external_contour, epsilon=epsilon)
    boundary = contour_to_boundary(simplified_contour)

    return ImagePreprocessingResult(
        mask=mask,
        external_contour=external_contour,
        simplified_contour=simplified_contour,
        boundary=boundary,
    )


def _extract_contours(mask: Mask) -> list[Contour]:
    _validate_mask(mask)
    boundary_edges = _build_boundary_edges(mask)
    return _stitch_edges_into_contours(boundary_edges)


def _validate_mask(mask: Mask) -> None:
    if not mask:
        raise ValueError("mask must not be empty")
    width = len(mask[0])
    if width == 0:
        raise ValueError("mask rows must not be empty")
    for row in mask:
        if len(row) != width:
            raise ValueError("mask must be rectangular")


def _build_boundary_edges(mask: Mask) -> list[Edge]:
    height = len(mask)
    width = len(mask[0])
    edges: list[Edge] = []

    for y in range(height):
        for x in range(width):
            if mask[y][x] == 0:
                continue

            if y == 0 or mask[y - 1][x] == 0:
                edges.append(((x, y), (x + 1, y)))
            if x == width - 1 or mask[y][x + 1] == 0:
                edges.append(((x + 1, y), (x + 1, y + 1)))
            if y == height - 1 or mask[y + 1][x] == 0:
                edges.append(((x + 1, y + 1), (x, y + 1)))
            if x == 0 or mask[y][x - 1] == 0:
                edges.append(((x, y + 1), (x, y)))

    return edges


def _stitch_edges_into_contours(edges: list[Edge]) -> list[Contour]:
    if not edges:
        return []

    start_to_ids: dict[Pixel, list[int]] = {}
    for edge_id, (start, _) in enumerate(edges):
        start_to_ids.setdefault(start, []).append(edge_id)

    visited: set[int] = set()
    contours: list[Contour] = []

    for edge_id in range(len(edges)):
        if edge_id in visited:
            continue

        contour: Contour = []
        start_edge_id = edge_id
        current_edge_id = edge_id
        max_steps = len(edges) + 1
        steps = 0

        while True:
            if current_edge_id in visited:
                break

            visited.add(current_edge_id)
            start, end = edges[current_edge_id]

            if not contour:
                contour.append(start)
            contour.append(end)

            steps += 1
            if steps > max_steps:
                raise RuntimeError("Contour stitching exceeded safety limit.")

            if current_edge_id != start_edge_id and end == contour[0]:
                break

            next_edge_id = _pick_next_edge(start_to_ids, edges, visited, end, start)
            if next_edge_id is None:
                break
            current_edge_id = next_edge_id

        if len(contour) >= 4 and contour[0] == contour[-1]:
            contour.pop()
            contours.append(contour)

    return contours


def _pick_next_edge(
    start_to_ids: dict[Pixel, list[int]],
    edges: list[Edge],
    visited: set[int],
    start_vertex: Pixel,
    prev_vertex: Pixel,
) -> int | None:
    edge_ids = start_to_ids.get(start_vertex, [])
    if not edge_ids:
        return None

    candidates = [edge_id for edge_id in edge_ids if edge_id not in visited]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    incoming = (start_vertex[0] - prev_vertex[0], start_vertex[1] - prev_vertex[1])
    ranked = sorted(
        candidates,
        key=lambda edge_id: _turn_score(incoming, _edge_direction(edges[edge_id])),
    )
    return ranked[0]


def _edge_direction(edge: Edge) -> Pixel:
    (x1, y1), (x2, y2) = edge
    return x2 - x1, y2 - y1


def _turn_score(incoming: Pixel, outgoing: Pixel) -> tuple[int, int]:
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    return (0 if cross < 0 else 1 if cross == 0 else 2, -dot)


def _polygon_area(contour: Contour) -> float:
    area = 0.0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _rdp(points: list[Pixel], epsilon: float) -> list[Pixel]:
    if len(points) <= 2:
        return points

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    index = -1

    for i in range(1, len(points) - 1):
        distance_to_line = _point_to_segment_distance(points[i], start, end)
        if distance_to_line > max_distance:
            max_distance = distance_to_line
            index = i

    if max_distance > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right

    return [start, end]


def _point_to_segment_distance(point: Pixel, segment_start: Pixel, segment_end: Pixel) -> float:
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end

    dx = x2 - x1
    dy = y2 - y1
    segment_length_sq = dx * dx + dy * dy

    if segment_length_sq <= _EPSILON:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    t = ((px - x1) * dx + (py - y1) * dy) / segment_length_sq
    t = max(0.0, min(1.0, t))
    projection = (x1 + t * dx, y1 + t * dy)
    return ((px - projection[0]) ** 2 + (py - projection[1]) ** 2) ** 0.5
