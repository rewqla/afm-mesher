from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities.point import Point
from src.domain.geometry.contour_pipeline import preprocess_contour, validate_contour

RawContour = list[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class BoundaryValidationIssue:
    contour_index: int
    kind: str
    message: str
    segment: tuple[tuple[float, float], tuple[float, float]] | None = None
    intersect_point: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class BoundaryValidationResult:
    is_valid: bool
    issues: list[BoundaryValidationIssue]

    @property
    def open_contours(self) -> list[BoundaryValidationIssue]:
        return [issue for issue in self.issues if issue.kind == "open_contour"]

    @property
    def self_intersections(self) -> list[BoundaryValidationIssue]:
        return [issue for issue in self.issues if issue.kind == "self_intersection"]


class BoundaryValidator:
    def __init__(self, closure_tolerance: float = 8.0, epsilon: float = 0.75) -> None:
        self._closure_tolerance = closure_tolerance
        self._epsilon = epsilon

    def validate(self, contours: list[RawContour]) -> BoundaryValidationResult:
        outer_idx = self._select_outer_contour_index(contours)
        if outer_idx is not None:
            return self._validate_single_contour(contours[outer_idx], contour_index=outer_idx)

        issues: list[BoundaryValidationIssue] = []
        for idx, contour in enumerate(contours):
            issues.extend(self._validate_contour_issues(contour, contour_index=idx))

        return BoundaryValidationResult(is_valid=not issues, issues=issues)

    def normalize_closed_contours(self, contours: list[RawContour]) -> list[RawContour]:
        normalized: list[RawContour] = []
        for contour in contours:
            processed = self._preprocess(contour)
            normalized.append([(point.x, point.y) for point in processed])
        return normalized

    def normalize_valid_closed_contours(self, contours: list[RawContour]) -> list[RawContour]:
        normalized: list[RawContour] = []
        for contour in contours:
            processed = self._preprocess(contour)
            result = validate_contour(processed)
            if result.is_closed and result.is_valid:
                normalized.append([(point.x, point.y) for point in processed])
        return normalized

    def select_outer_contour(self, contours: list[RawContour]) -> RawContour | None:
        closed_contours = self.normalize_valid_closed_contours(contours)
        if not closed_contours:
            return None
        return max(closed_contours, key=lambda contour: abs(self._signed_area(contour)))

    def _preprocess(self, contour: RawContour) -> list[Point]:
        points = [Point(float(x), float(y)) for x, y in contour]
        return preprocess_contour(points, epsilon=self._epsilon, closure_tolerance=self._closure_tolerance)

    def _validate_single_contour(self, contour: RawContour, *, contour_index: int) -> BoundaryValidationResult:
        issues = self._validate_contour_issues(contour, contour_index=contour_index)
        return BoundaryValidationResult(is_valid=not issues, issues=issues)

    def _validate_contour_issues(self, contour: RawContour, *, contour_index: int) -> list[BoundaryValidationIssue]:
        processed = self._preprocess(contour)
        if len(processed) < 3:
            if len(processed) >= 2:
                return [
                    BoundaryValidationIssue(
                        contour_index=contour_index,
                        kind="open_contour",
                        message="Contour is not closed.",
                        segment=((processed[0].x, processed[0].y), (processed[-1].x, processed[-1].y)),
                    )
                ]
            return [
                BoundaryValidationIssue(
                    contour_index=contour_index,
                    kind="too_few_points",
                    message="Contour must contain at least 3 points.",
                )
            ]

        result = validate_contour(processed)
        if not result.is_closed:
            return [
                BoundaryValidationIssue(
                    contour_index=contour_index,
                    kind="open_contour",
                    message="Contour is not closed.",
                    segment=((processed[0].x, processed[0].y), (processed[-1].x, processed[-1].y)),
                )
            ]

        if not result.is_valid:
            intersect_point = None
            if result.intersect_point is not None:
                intersect_point = (result.intersect_point.x, result.intersect_point.y)
            return [
                BoundaryValidationIssue(
                    contour_index=contour_index,
                    kind="self_intersection",
                    message="Contour has self-intersections.",
                    intersect_point=intersect_point,
                )
            ]

        return []

    def _select_outer_contour_index(self, contours: list[RawContour]) -> int | None:
        best_idx: int | None = None
        best_area = float("-inf")
        for idx, contour in enumerate(contours):
            processed = self._preprocess(contour)
            if len(processed) < 3:
                continue
            result = validate_contour(processed)
            if not result.is_closed:
                continue

            normalized = [(point.x, point.y) for point in processed]
            area = abs(self._signed_area(normalized))
            if area > best_area:
                best_area = area
                best_idx = idx
        return best_idx

    def _signed_area(self, contour: RawContour) -> float:
        area = 0.0
        for idx in range(len(contour) - 1):
            x1, y1 = contour[idx]
            x2, y2 = contour[idx + 1]
            area += x1 * y2 - x2 * y1
        return area / 2.0
