"""Arc-length resampling of drawn contours to the meshing step.

Drawn / image contours have vertices every 1-3 px along curves. The mesher only splits long sides,
so such vertices become boundary nodes and produce tiny, needle-like triangles. Resampling keeps
corners (sharp turns) and anchor points (e.g. where a cut ends on the contour) exactly and
replaces the dense vertices between them with points spaced about `step` along the curve.
"""
from __future__ import annotations

from bisect import bisect_right
from math import acos, degrees, hypot

from src.domain.entities.point import Point

# A vertex is a corner when the contour turns by more than this over the corner window.
CORNER_ANGLE_DEG = 30.0


def resample_polyline(
        points: list[Point],
        step: float,
        closed: bool,
        anchors: list[tuple[Point, float, bool]] = (),
        max_deviation: float = 1.0,
        corner_angle_deg: float = CORNER_ANGLE_DEG,
) -> list[Point]:
    """Points about `step` apart along the polyline (no closing duplicate for closed rings).

    Kept exactly: the ends of an open polyline, corners, and every anchor (point, snap distance,
    exact) lying within its snap distance of the polyline — the anchor itself when `exact` (a nearby
    vertex is moved onto it), otherwise its projection onto the polyline. Between kept vertices the
    spacing is reduced, if needed, until the original vertices are within `max_deviation` of the result.
    """
    pts = list(points)
    if closed and len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    min_count = 3 if closed else 2
    if len(pts) < min_count or step <= 0:
        return list(points)

    pts, fixed = _insert_anchors(pts, closed, anchors)
    n = len(pts)
    seg_count = n if closed else n - 1
    lengths = [hypot(pts[(i + 1) % n].x - pts[i].x, pts[(i + 1) % n].y - pts[i].y) for i in range(seg_count)]
    arc = [0.0]
    for length in lengths:
        arc.append(arc[-1] + length)
    total = arc[-1]
    if total <= 1e-9:
        return list(points)

    if not closed:
        fixed.update((0, n - 1))
    fixed.update(_corners(pts, arc, closed, step, corner_angle_deg))
    if not fixed:
        fixed.add(0)
    kept = sorted(fixed)

    spans: list[tuple[int, int]] = []  # (start vertex, end vertex); end may wrap for closed rings
    for idx, start in enumerate(kept):
        if idx + 1 < len(kept):
            spans.append((start, kept[idx + 1]))
        elif closed:
            spans.append((start, kept[0] + n))
    counts = [_span_count(pts, arc, total, closed, a, b, step, max_deviation) for a, b in spans]

    def size() -> int:
        return sum(counts) + (0 if closed else 1)

    min_points = min(len(pts), 6 if closed else 2)
    while size() < min_points:
        longest = max(range(len(spans)), key=lambda i: _span_length(arc, total, *spans[i]) / counts[i])
        counts[longest] += 1

    result: list[Point] = []
    for (a, b), count in zip(spans, counts):
        if count >= b - a and _min_side(pts, a, b) >= 0.5 * step:
            # Already no denser than the step (long sides are split later by the mesher): keep as is.
            result.extend(pts[k % n] for k in range(a, b))
            continue
        result.append(pts[a % n])
        start_s = _arc_position(arc, total, a)
        span_len = _span_length(arc, total, a, b)
        for j in range(1, count):
            result.append(_point_at(pts, arc, total, closed, start_s + span_len * j / count))
    if not closed:
        result.append(pts[-1])
    return result


def _insert_anchors(
        pts: list[Point],
        closed: bool,
        anchors: list[tuple[Point, float, bool]],
) -> tuple[list[Point], set[int]]:
    n = len(pts)
    seg_count = n if closed else n - 1
    fixed_vertices: set[int] = set()
    inserts: dict[int, list[tuple[float, Point]]] = {}
    if not anchors:
        return pts, fixed_vertices
    # Uniform grid over segment boxes: each anchor only looks at segments near it.
    max_snap = max(snap for _, snap, _ in anchors)
    cell = max(1.0, 2.0 * max_snap, 4.0 * sum(
        hypot(pts[(i + 1) % n].x - pts[i].x, pts[(i + 1) % n].y - pts[i].y) for i in range(seg_count)
    ) / max(seg_count, 1))
    grid: dict[tuple[int, int], list[int]] = {}
    for i in range(seg_count):
        a = pts[i]
        b = pts[(i + 1) % n]
        for gx in range(int(min(a.x, b.x) // cell), int(max(a.x, b.x) // cell) + 1):
            for gy in range(int(min(a.y, b.y) // cell), int(max(a.y, b.y) // cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    replaced: dict[int, Point] = {}
    for anchor, snap, exact in anchors:
        best: tuple[float, int, float] | None = None
        nearby: set[int] = set()
        for gx in range(int((anchor.x - snap) // cell), int((anchor.x + snap) // cell) + 1):
            for gy in range(int((anchor.y - snap) // cell), int((anchor.y + snap) // cell) + 1):
                nearby.update(grid.get((gx, gy), ()))
        for i in sorted(nearby):
            a = pts[i]
            b = pts[(i + 1) % n]
            if (
                    anchor.x < min(a.x, b.x) - snap or anchor.x > max(a.x, b.x) + snap
                    or anchor.y < min(a.y, b.y) - snap or anchor.y > max(a.y, b.y) + snap
            ):
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            denom = dx * dx + dy * dy
            t = 0.0 if denom <= 1e-18 else max(0.0, min(1.0, ((anchor.x - a.x) * dx + (anchor.y - a.y) * dy) / denom))
            d = hypot(a.x + t * dx - anchor.x, a.y + t * dy - anchor.y)
            if d <= snap and (best is None or d < best[0]):
                best = (d, i, t)
        if best is None:
            continue
        d, i, t = best
        a = pts[i]
        b = pts[(i + 1) % n]
        j = (i + 1) % n
        length = hypot(b.x - a.x, b.y - a.y)
        if exact:
            # The anchor itself becomes a vertex; a vertex within the snap distance is moved onto it
            # (inserting next to it would leave a sub-pixel side).
            if hypot(a.x - anchor.x, a.y - anchor.y) <= snap:
                replaced[i] = anchor
                fixed_vertices.add(i)
            elif hypot(b.x - anchor.x, b.y - anchor.y) <= snap:
                replaced[j] = anchor
                fixed_vertices.add(j)
            else:
                inserts.setdefault(i, []).append((t, anchor))
        # Within 1e-6 px of a vertex: that vertex is the anchor.
        elif t * length <= 1e-6:
            fixed_vertices.add(i)
        elif (1.0 - t) * length <= 1e-6:
            fixed_vertices.add(j)
        else:
            point = anchor if d <= 1e-6 else Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            inserts.setdefault(i, []).append((t, point))
    if not inserts and not replaced:
        return pts, fixed_vertices

    out: list[Point] = []
    fixed: set[int] = set()
    for i, p in enumerate(pts):
        p = replaced.get(i, p)
        if i in fixed_vertices:
            fixed.add(len(out))
        out.append(p)
        for _, point in sorted(inserts.get(i, ()), key=lambda item: item[0]):
            if point != out[-1]:
                fixed.add(len(out))
                out.append(point)
    return out, fixed


def _corners(pts: list[Point], arc: list[float], closed: bool, step: float, angle_deg: float) -> set[int]:
    """Vertices where the direction over a short window (a few px) turns by more than angle_deg;
    of neighbouring candidates only the sharpest one is kept."""
    n = len(pts)
    total = arc[-1]
    window = max(3.0, min(step * 0.3, 8.0))
    turns: list[tuple[float, int]] = []
    for i in range(n):
        s = arc[i]
        if not closed and (s - window < 0.0 or s + window > total):
            continue
        back = _point_at(pts, arc, total, closed, s - window)
        ahead = _point_at(pts, arc, total, closed, s + window)
        p = pts[i]
        v1 = (p.x - back.x, p.y - back.y)
        v2 = (ahead.x - p.x, ahead.y - p.y)
        norm = hypot(*v1) * hypot(*v2)
        if norm <= 1e-12:
            continue
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / norm))
        turn = degrees(acos(cos))
        if turn > angle_deg:
            turns.append((turn, i))
    corners: list[int] = []
    for turn, i in sorted(turns, reverse=True):
        if all(_arc_gap(arc, total, closed, i, j) > window for j in corners):
            corners.append(i)
    return set(corners)


def _arc_gap(arc: list[float], total: float, closed: bool, i: int, j: int) -> float:
    gap = abs(arc[i] - arc[j])
    return min(gap, total - gap) if closed else gap


def _min_side(pts: list[Point], a: int, b: int) -> float:
    n = len(pts)
    return min(
        hypot(pts[(k + 1) % n].x - pts[k % n].x, pts[(k + 1) % n].y - pts[k % n].y) for k in range(a, b)
    )


def _arc_position(arc: list[float], total: float, k: int) -> float:
    """Arc length at vertex k; k past the last vertex of a closed ring continues around it."""
    last = len(arc) - 1
    return arc[k] if k <= last else arc[k - last] + total


def _span_length(arc: list[float], total: float, a: int, b: int) -> float:
    return _arc_position(arc, total, b) - _arc_position(arc, total, a)


def _span_count(
        pts: list[Point],
        arc: list[float],
        total: float,
        closed: bool,
        a: int,
        b: int,
        step: float,
        max_deviation: float,
) -> int:
    n = len(pts)
    span_len = _span_length(arc, total, a, b)
    original = b - a
    count = max(1, round(span_len / step))
    start_s = arc[a]
    inner = [pts[k % n] for k in range(a + 1, b)]
    while count < original:
        chords = [_point_at(pts, arc, total, closed, start_s + span_len * j / count) for j in range(count + 1)]
        if all(_distance_to_chords(p, chords) <= max_deviation for p in inner):
            break
        count += 1
    return count


def _distance_to_chords(p: Point, chords: list[Point]) -> float:
    best = float("inf")
    for a, b in zip(chords, chords[1:]):
        dx = b.x - a.x
        dy = b.y - a.y
        denom = dx * dx + dy * dy
        t = 0.0 if denom <= 1e-18 else max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / denom))
        best = min(best, hypot(a.x + t * dx - p.x, a.y + t * dy - p.y))
    return best


def _point_at(pts: list[Point], arc: list[float], total: float, closed: bool, s: float) -> Point:
    n = len(pts)
    if closed:
        s %= total
    else:
        s = max(0.0, min(total, s))
    i = min(bisect_right(arc, s) - 1, len(arc) - 2)
    i = max(i, 0)
    seg = arc[i + 1] - arc[i]
    t = 0.0 if seg <= 1e-12 else (s - arc[i]) / seg
    a = pts[i]
    b = pts[(i + 1) % n]
    return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
