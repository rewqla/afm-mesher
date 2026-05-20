from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.domain.entities.point import Point
from src.domain.geometry.geometry_utils import orientation, segments_intersect

Vector2 = Point
_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class ValidationResult:
    is_valid: bool
    is_closed: bool
    intersect_point: Vector2 | None = None


def preprocess_contour(
    raw_points: list[Vector2],
    epsilon: float,
    closure_tolerance: float,
) -> list[Vector2]:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if closure_tolerance <= 0:
        raise ValueError("closure_tolerance must be > 0")
    if not raw_points:
        return []

    explicitly_closed = len(raw_points) >= 3 and _distance(raw_points[0], raw_points[-1]) <= closure_tolerance
    working_points = list(raw_points[:-1] if explicitly_closed else raw_points)

    cleaned = _remove_consecutive_duplicates(working_points, epsilon)
    if len(cleaned) <= 2:
        if explicitly_closed and len(cleaned) >= 2:
            return [*cleaned, cleaned[0]]
        return cleaned

    simplified = _ramer_douglas_peucker(cleaned, epsilon)
    simplified = _remove_consecutive_duplicates(simplified, epsilon)
    if explicitly_closed:
        if len(simplified) < 3:
            return [*cleaned, cleaned[0]]
        return [*simplified, simplified[0]]
    return _clip_tail_and_snap_closure(simplified, closure_tolerance)


def validate_contour(processed_points: list[Vector2]) -> ValidationResult:
    if len(processed_points) < 3:
        return ValidationResult(is_valid=False, is_closed=False, intersect_point=None)

    is_closed = _is_closed(processed_points)
    if not is_closed:
        return ValidationResult(is_valid=False, is_closed=False, intersect_point=None)

    points = _repair_closure_micro_loop(processed_points)
    intersection = _find_self_intersection(points)
    if intersection is None:
        return ValidationResult(is_valid=True, is_closed=True, intersect_point=None)

    repaired = _extract_largest_loop(points, intersection)
    repaired_intersection = _find_self_intersection(repaired)
    return ValidationResult(
        is_valid=repaired_intersection is None,
        is_closed=True,
        intersect_point=repaired_intersection,
    )


def smart_append_contour(
    existing_contour: list[Vector2],
    new_stroke: list[Vector2],
    snap_tolerance: float,
    attach_to: str = "end",
) -> list[Vector2]:
    if not existing_contour:
        return list(new_stroke)
    if not new_stroke:
        return list(existing_contour)
    if snap_tolerance <= 0:
        raise ValueError("snap_tolerance must be > 0")

    target = list(existing_contour)
    stroke = list(new_stroke)

    if attach_to == "start":
        anchor = target[0]
        direction = _head_direction(target)
        aligned = _align_stroke_to_direction(anchor, stroke, Point(-direction.x, -direction.y))
        prefix = list(reversed(aligned))
        return _remove_consecutive_duplicates([*prefix, *target], snap_tolerance * 0.25)

    anchor = target[-1]
    direction = _tail_direction(target)
    aligned = _align_stroke_to_direction(anchor, stroke, direction)
    return _remove_consecutive_duplicates([*target, *aligned], snap_tolerance * 0.25)


def PreprocessContour(
    rawPoints: list[Vector2],
    epsilon: float,
    closureTolerance: float,
) -> list[Vector2]:
    return preprocess_contour(rawPoints, epsilon, closureTolerance)


def ValidateContour(processedPoints: list[Vector2]) -> ValidationResult:
    return validate_contour(processedPoints)


def SmartAppendContour(
    existingContour: list[Vector2],
    newStroke: list[Vector2],
    snapTolerance: float,
    attachTo: str = "end",
) -> list[Vector2]:
    return smart_append_contour(existingContour, newStroke, snapTolerance, attach_to=attachTo)


def _remove_consecutive_duplicates(points: list[Vector2], epsilon: float) -> list[Vector2]:
    cleaned: list[Vector2] = [points[0]]
    for point in points[1:]:
        if _distance(cleaned[-1], point) >= epsilon:
            cleaned.append(point)
    if len(cleaned) >= 2 and _distance(cleaned[0], cleaned[-1]) < epsilon:
        cleaned[-1] = cleaned[0]
    return cleaned


def _ramer_douglas_peucker(points: list[Vector2], epsilon: float) -> list[Vector2]:
    if len(points) <= 2:
        return list(points)

    max_distance = -1.0
    split_index = -1
    start = points[0]
    end = points[-1]

    for index in range(1, len(points) - 1):
        distance = _point_to_segment_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = index

    if max_distance <= epsilon:
        return [start, end]

    left = _ramer_douglas_peucker(points[: split_index + 1], epsilon)
    right = _ramer_douglas_peucker(points[split_index:], epsilon)
    return [*left[:-1], *right]


def _clip_tail_and_snap_closure(points: list[Vector2], closure_tolerance: float) -> list[Vector2]:
    if len(points) < 3:
        return points

    first = points[0]
    closure_index: int | None = None
    for index in range(len(points) - 1, 2, -1):
        if _distance(first, points[index]) <= closure_tolerance:
            closure_index = index
            break

    if closure_index is None:
        return points

    clipped = list(points[: closure_index + 1])
    clipped[-1] = clipped[0]
    return clipped


def _is_closed(points: list[Vector2]) -> bool:
    return len(points) >= 4 and _distance(points[0], points[-1]) <= _EPSILON


def _find_self_intersection(points: list[Vector2]) -> Vector2 | None:
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    for i, (a1, a2) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if _segments_are_topologically_adjacent(i, j, len(segments)):
                if _is_degenerate_turn(points, i, j):
                    return points[j]
                continue

            b1, b2 = segments[j]
            if segments_intersect(a1, a2, b1, b2, include_endpoints=True):
                return _segment_intersection_point(a1, a2, b1, b2)
    return None


def _segments_are_topologically_adjacent(i: int, j: int, segment_count: int) -> bool:
    if j == i + 1:
        return True
    return i == 0 and j == segment_count - 1


def _is_degenerate_turn(points: list[Vector2], i: int, j: int) -> bool:
    if j != i + 1:
        return False
    a = points[i]
    b = points[i + 1]
    c = points[j + 1]
    if orientation(a, b, c) != 0:
        return False
    ab = Point(b.x - a.x, b.y - a.y)
    bc = Point(c.x - b.x, c.y - b.y)
    return ab.x * bc.x + ab.y * bc.y < 0


def _repair_closure_micro_loop(points: list[Vector2]) -> list[Vector2]:
    if len(points) < 5:
        return points
    first = points[0]
    for index in range(1, min(len(points) - 1, 6)):
        if _distance(first, points[index]) <= _EPSILON:
            repaired = [first, *points[index + 1 :]]
            repaired[-1] = repaired[0]
            return repaired
    return points


def _extract_largest_loop(points: list[Vector2], intersection: Vector2) -> list[Vector2]:
    matching_indices = [idx for idx, point in enumerate(points[:-1]) if _distance(point, intersection) <= 1.0]
    if len(matching_indices) < 2:
        return points

    best_loop = points
    best_area = abs(_signed_area(points))
    for start_idx in matching_indices:
        for end_idx in matching_indices:
            if end_idx <= start_idx + 1:
                continue
            candidate = [*points[start_idx : end_idx + 1]]
            candidate[-1] = candidate[0]
            if len(candidate) < 4:
                continue
            area = abs(_signed_area(candidate))
            if area > best_area:
                best_area = area
                best_loop = candidate
    return best_loop


def _segment_intersection_point(a1: Vector2, a2: Vector2, b1: Vector2, b2: Vector2) -> Vector2:
    denominator = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x)
    if abs(denominator) <= _EPSILON:
        return b1
    det_a = a1.x * a2.y - a1.y * a2.x
    det_b = b1.x * b2.y - b1.y * b2.x
    px = (det_a * (b1.x - b2.x) - (a1.x - a2.x) * det_b) / denominator
    py = (det_a * (b1.y - b2.y) - (a1.y - a2.y) * det_b) / denominator
    return Point(px, py)


def _tail_direction(points: list[Vector2]) -> Vector2:
    if len(points) < 2:
        return Point(1.0, 0.0)
    a = points[-2]
    b = points[-1]
    dx = b.x - a.x
    dy = b.y - a.y
    length = sqrt(dx * dx + dy * dy)
    if length <= _EPSILON:
        return Point(1.0, 0.0)
    return Point(dx / length, dy / length)


def _head_direction(points: list[Vector2]) -> Vector2:
    if len(points) < 2:
        return Point(1.0, 0.0)
    a = points[0]
    b = points[1]
    dx = b.x - a.x
    dy = b.y - a.y
    length = sqrt(dx * dx + dy * dy)
    if length <= _EPSILON:
        return Point(1.0, 0.0)
    return Point(dx / length, dy / length)


def _align_stroke_to_direction(
    anchor: Vector2,
    stroke: list[Vector2],
    direction: Vector2,
) -> list[Vector2]:
    if not stroke:
        return []

    aligned: list[Vector2] = [anchor]
    offsets = [Point(point.x - stroke[0].x, point.y - stroke[0].y) for point in stroke]

    for index, offset in enumerate(offsets[1:], start=1):
        point = Point(anchor.x + offset.x, anchor.y + offset.y)
        if index <= 2:
            projection = offset.x * direction.x + offset.y * direction.y
            if projection < 0:
                projection = 0.0
            point = Point(anchor.x + direction.x * projection, anchor.y + direction.y * projection)
        aligned.append(point)
    return aligned


def _point_to_segment_distance(point: Vector2, start: Vector2, end: Vector2) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if abs(dx) <= _EPSILON and abs(dy) <= _EPSILON:
        return _distance(point, start)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return _distance(point, projection)


def _distance(a: Vector2, b: Vector2) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)


def _signed_area(points: list[Vector2]) -> float:
    area = 0.0
    for idx in range(len(points) - 1):
        p1 = points[idx]
        p2 = points[idx + 1]
        area += p1.x * p2.y - p2.x * p1.y
    return area / 2.0
