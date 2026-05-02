from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.domain.entities.point import Point

Mask = list[list[int]]
Pixel = tuple[int, int]
Contour = list[Pixel]
Edge = tuple[Pixel, Pixel]


@dataclass(slots=True)
class ImageBoundaryData:
    mask: Mask
    contours: list[Contour]
    boundaries: list[list[Point]]


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


def extract_contours(mask: Mask) -> list[Contour]:
    _validate_mask(mask)
    boundary_edges = _build_boundary_edges(mask)
    return _stitch_edges_into_contours(boundary_edges)


def contours_to_boundaries(contours: list[Contour]) -> list[list[Point]]:
    return [[Point(float(x), float(y)) for x, y in contour] for contour in contours]


def process_image_to_boundaries(
    image_path: str | Path,
    threshold: int = 127,
    invert: bool = False,
) -> ImageBoundaryData:
    mask = load_binary_mask(image_path=image_path, threshold=threshold, invert=invert)
    contours = extract_contours(mask)
    boundaries = contours_to_boundaries(contours)
    return ImageBoundaryData(mask=mask, contours=contours, boundaries=boundaries)


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

            # Top side
            if y == 0 or mask[y - 1][x] == 0:
                edges.append(((x, y), (x + 1, y)))
            # Right side
            if x == width - 1 or mask[y][x + 1] == 0:
                edges.append(((x + 1, y), (x + 1, y + 1)))
            # Bottom side
            if y == height - 1 or mask[y + 1][x] == 0:
                edges.append(((x + 1, y + 1), (x, y + 1)))
            # Left side
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
    # Prefer right turns, then straight, then left turns for clockwise traversal.
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    return (0 if cross < 0 else 1 if cross == 0 else 2, -dot)
