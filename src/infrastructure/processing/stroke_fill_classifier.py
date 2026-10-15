"""Split an image mask into filled regions and pen strokes.

A filled shape (an inclusion drawn with fill, a solid region) must become a closed contour, while a
pen stroke (a line about the pen width) is a cut or, when closed, a boundary drawn by its centerline.
A connected component is a fill when some of its pixels survive `max_stroke_halfwidth + 1` erosions
by a 3x3 square, i.e. it is somewhere at least 2 * (max_stroke_halfwidth + 1) + 1 px wide; otherwise
it is a stroke. Strokes touching a fill are part of that fill's component and stay with it.
"""
from __future__ import annotations

from collections import deque
from math import hypot

import numpy as np

from src.infrastructure.processing.image_boundary_extractor import extract_contours
from src.infrastructure.processing.stroke_centerline_extractor import (
    _NEIGHBORS_8,
    _trace_component,
    _zhang_suen_thinning,
)

Mask = list[list[int]]
Pixel = tuple[int, int]
Contour = list[Pixel]

# Half-width (px) up to which a component counts as a stroke: lines up to ~9 px wide. The canvas pen
# is 2 px; lines in loaded pictures are a few px.
MAX_STROKE_HALFWIDTH = 4
# Skeleton pieces up to this many pixels between two junctions belong to one crossing.
_JUNCTION_MERGE_PX = 8
# Distance (skeleton pixels) from a junction at which a branch direction is measured.
_DIRECTION_PROBE_PX = 12


def split_fill_and_strokes(mask: Mask, max_stroke_halfwidth: int = MAX_STROKE_HALFWIDTH) -> tuple[np.ndarray, np.ndarray]:
    """(fill, strokes) boolean arrays: components with a thick core, and the rest of the mask."""
    filled = np.asarray(mask) != 0
    if not filled.any():
        return filled, filled.copy()
    core = filled
    for _ in range(max_stroke_halfwidth + 1):
        core = _erode(core)
        if not core.any():
            return np.zeros_like(filled), filled
    # Grow the cores through their components (8-connected); only the thin remainder is walked.
    fill = core.copy()
    thin = filled & ~core
    height, width = filled.shape
    seeds = np.argwhere(thin & _dilate(core))
    queue: deque[tuple[int, int]] = deque((int(y), int(x)) for y, x in seeds)
    for y, x in queue:
        fill[y, x] = True
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                if 0 <= nx < width and thin[ny, nx] and not fill[ny, nx]:
                    fill[ny, nx] = True
                    queue.append((ny, nx))
    return fill, filled & ~fill


def extract_fill_and_stroke_contours(mask: Mask, max_stroke_halfwidth: int = MAX_STROKE_HALFWIDTH) -> list[Contour]:
    """Closed boundary contours of the filled regions, then centerlines of the strokes."""
    fill, strokes = split_fill_and_strokes(mask, max_stroke_halfwidth)
    contours: list[Contour] = []
    if fill.any():
        contours.extend(extract_contours(fill.astype(np.int64).tolist()))
    if strokes.any():
        contours.extend(stroke_centerlines(strokes))
    return contours


def stroke_centerlines(strokes: np.ndarray) -> list[Contour]:
    """Centerlines of the stroke pixels, one polyline per drawn line: crossing / touching strokes are
    split at the junctions of the skeleton and rejoined along the straightest continuation.
    Thinning runs on the bounding box of the strokes only (it is pure Python)."""
    ys, xs = np.nonzero(strokes)
    y0 = max(int(ys.min()) - 1, 0)
    x0 = max(int(xs.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, strokes.shape[0])
    x1 = min(int(xs.max()) + 2, strokes.shape[1])
    crop = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=np.int64)
    crop[1:-1, 1:-1] = strokes[y0:y1, x0:x1]
    skeleton = {(x, y) for y, row in enumerate(_zhang_suen_thinning(crop.tolist())) for x, v in enumerate(row) if v}
    lines = _skeleton_polylines(skeleton)
    return [[(x + x0 - 1, y + y0 - 1) for x, y in line] for line in lines]


def _skeleton_polylines(skeleton: set[Pixel]) -> list[Contour]:
    def neighbours(p: Pixel, pixels: set[Pixel]) -> list[Pixel]:
        return [(p[0] + dx, p[1] + dy) for dx, dy in _NEIGHBORS_8 if (p[0] + dx, p[1] + dy) in pixels]

    junction_pixels = {p for p in skeleton if len(neighbours(p, skeleton)) >= 3}
    if not junction_pixels:
        return [c for c in (_trace_component(comp) for comp in _components(skeleton)) if len(c) >= 2]

    # Thinning turns one crossing of two strokes into two junctions joined by a short piece: such
    # pieces (both ends at junctions, at most _JUNCTION_MERGE_PX long) are part of the junction.
    for comp in _components(skeleton - junction_pixels):
        if len(comp) <= _JUNCTION_MERGE_PX:
            touching = {q for p in comp for q in neighbours(p, junction_pixels)}
            line = _trace_component(comp)
            ends_touch = [any(q in touching for q in neighbours(end, junction_pixels)) for end in (line[0], line[-1])]
            if all(ends_touch) and len(_components(touching)) >= 2:
                junction_pixels |= comp

    # Junction clusters (adjacent junction pixels) and their centre pixels.
    cluster_of: dict[Pixel, int] = {}
    centres: list[Pixel] = []
    for comp in _components(junction_pixels):
        cx = round(sum(p[0] for p in comp) / len(comp))
        cy = round(sum(p[1] for p in comp) / len(comp))
        for p in comp:
            cluster_of[p] = len(centres)
        centres.append((cx, cy))

    # Branches between junctions: (polyline, junction at start or None, junction at end or None).
    branches: list[tuple[Contour, int | None, int | None]] = []
    for comp in _components(skeleton - junction_pixels):
        line = _trace_component(comp)
        if len(line) >= 2 and line[0] == line[-1]:
            branches.append((line, None, None))  # closed loop not touching any junction
            continue

        def junction_at(p: Pixel, other: int | None = None) -> int | None:
            ids = sorted({cluster_of[q] for q in neighbours(p, junction_pixels)} - {other})
            return ids[0] if ids else None

        start = junction_at(line[0])
        end = junction_at(line[-1], other=start if len(line) == 1 else None)
        line = ([centres[start]] if start is not None else []) + line + ([centres[end]] if end is not None else [])
        # Short spurs of the thinning (a few px from a junction to a free end) are dropped.
        if (start is None) != (end is None) and len(line) <= 4:
            continue
        branches.append((line, start, end))

    # At every junction pair the branch ends that continue each other most straightly.
    ends_at: dict[int, list[tuple[int, int]]] = {}
    for b, (_, start, end) in enumerate(branches):
        if start is not None:
            ends_at.setdefault(start, []).append((b, 0))
        if end is not None:
            ends_at.setdefault(end, []).append((b, 1))
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for junction, ends in ends_at.items():
        centre = centres[junction]

        def outward(end: tuple[int, int]) -> tuple[float, float]:
            line = branches[end[0]][0]
            k = _DIRECTION_PROBE_PX
            probe = line[min(k, len(line) - 1)] if end[1] == 0 else line[max(len(line) - 1 - k, 0)]
            dx, dy = probe[0] - centre[0], probe[1] - centre[1]
            norm = max(hypot(dx, dy), 1e-9)
            return dx / norm, dy / norm

        free = list(ends)
        if len(free) == 2 and free[0][0] != free[1][0]:
            partner[free[0]], partner[free[1]] = free[1], free[0]
            continue
        pairs = []
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                if free[i][0] == free[j][0]:
                    continue
                u, v = outward(free[i]), outward(free[j])
                pairs.append((u[0] * v[0] + u[1] * v[1], free[i], free[j]))
        used: set[tuple[int, int]] = set()
        for cos, a, b in sorted(pairs):
            if cos > -0.85:  # straighter than ~150 degrees only
                break
            if a in used or b in used:
                continue
            used.update((a, b))
            partner[a], partner[b] = b, a

    # Walk chains of paired branches.
    visited: set[int] = set()
    polylines: list[Contour] = []

    def walk(b: int, entry: int) -> Contour:
        chain: Contour = []
        while b not in visited:
            visited.add(b)
            line = branches[b][0] if entry == 0 else branches[b][0][::-1]
            chain.extend(line if not chain else line[1:])
            exit_end = (b, 1 - entry)
            nxt = partner.get(exit_end)
            if nxt is None:
                break
            b, entry = nxt
        return chain

    for b, (_, start, end) in enumerate(branches):
        if b in visited:
            continue
        if (b, 0) not in partner:
            polylines.append(walk(b, 0))
        elif (b, 1) not in partner:
            polylines.append(walk(b, 1))
    for b in range(len(branches)):
        if b not in visited:  # closed chains through junctions
            chain = walk(b, 0)
            if chain[0] != chain[-1]:
                chain.append(chain[0])
            polylines.append(chain)
    return [p for p in polylines if len(p) >= 2]


def _components(pixels: set[Pixel]) -> list[set[Pixel]]:
    remaining = set(pixels)
    out: list[set[Pixel]] = []
    for seed in sorted(pixels, key=lambda p: (p[1], p[0])):
        if seed not in remaining:
            continue
        comp = {seed}
        remaining.discard(seed)
        queue = deque([seed])
        while queue:
            x, y = queue.popleft()
            for dx, dy in _NEIGHBORS_8:
                q = (x + dx, y + dy)
                if q in remaining:
                    remaining.discard(q)
                    comp.add(q)
                    queue.append(q)
        out.append(comp)
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    h, w = mask.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    h, w = mask.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy:dy + h, dx:dx + w]
    return out
