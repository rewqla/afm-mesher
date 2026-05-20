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
        issues: list[BoundaryValidationIssue] = []
        for idx, contour in enumerate(contours):
            processed = self._preprocess(contour)
            if len(processed) < 3:
                if len(processed) >= 2:
                    issues.append(
                        BoundaryValidationIssue(
                            contour_index=idx,
                            kind="open_contour",
                            message="Contour is not closed.",
                            segment=((processed[0].x, processed[0].y), (processed[-1].x, processed[-1].y)),
                        )
                    )
                    continue
                issues.append(
                    BoundaryValidationIssue(
                        contour_index=idx,
                        kind="too_few_points",
                        message="Contour must contain at least 3 points.",
                    )
                )
                continue

            result = validate_contour(processed)
            if not result.is_closed:
                issues.append(
                    BoundaryValidationIssue(
                        contour_index=idx,
                        kind="open_contour",
                        message="Contour is not closed.",
                        segment=((processed[0].x, processed[0].y), (processed[-1].x, processed[-1].y)),
                    )
                )
                continue

            if not result.is_valid:
                intersect_point = None
                if result.intersect_point is not None:
                    intersect_point = (result.intersect_point.x, result.intersect_point.y)
                issues.append(
                    BoundaryValidationIssue(
                        contour_index=idx,
                        kind="self_intersection",
                        message="Contour has self-intersections.",
                        intersect_point=intersect_point,
                    )
                )

        return BoundaryValidationResult(is_valid=not issues, issues=issues)

    def normalize_closed_contours(self, contours: list[RawContour]) -> list[RawContour]:
        normalized: list[RawContour] = []
        for contour in contours:
            processed = self._preprocess(contour)
            normalized.append([(point.x, point.y) for point in processed])
        return normalized

    def _preprocess(self, contour: RawContour) -> list[Point]:
        points = [Point(float(x), float(y)) for x, y in contour]
        return preprocess_contour(points, epsilon=self._epsilon, closure_tolerance=self._closure_tolerance)
