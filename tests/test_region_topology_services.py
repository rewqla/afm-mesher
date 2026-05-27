import unittest
from unittest.mock import patch
import random

import _bootstrap  # noqa: F401
from src.application.services.boundary_detection_service import BoundaryDetectionService
from src.application.services.obstacle_processor import ObstacleProcessor
from src.application.services.region_classifier import RegionClassifier
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import point_in_polygon, segments_intersect

try:
    from shapely.geometry import Polygon

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    Polygon = None  # type: ignore[assignment]
    _HAS_SHAPELY = False


class TestRegionTopologyServices(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary_detection = BoundaryDetectionService()
        self.classifier = RegionClassifier()
        self.obstacles = ObstacleProcessor()

    def test_classifies_shell_hole_and_nested_island(self) -> None:
        outer = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)]
        island = [(7, 7), (9, 7), (9, 9), (7, 9), (7, 7)]
        polygons = self.boundary_detection.detect([outer, hole, island])

        regions = self.classifier.classify(polygons)
        self.assertEqual(len(regions), 2)

        largest = max(regions, key=lambda region: len(region.shell.points))
        self.assertEqual(len(largest.holes), 1)

    def test_classifies_all_direct_inner_polygons_as_holes(self) -> None:
        outer = [(0, 0), (30, 0), (30, 20), (0, 20), (0, 0)]
        holes = [
            [(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)],
            [(12, 4), (16, 4), (16, 8), (12, 8), (12, 4)],
            [(20, 4), (24, 4), (24, 8), (20, 8), (20, 4)],
        ]
        polygons = self.boundary_detection.detect([outer, *holes])

        regions = self.classifier.classify(polygons)
        main_region = max(regions, key=lambda region: abs(self.classifier._signed_area(region.shell.points)))

        self.assertEqual(len(main_region.holes), 3)

    def test_filters_triangles_inside_hole(self) -> None:
        hole = self.boundary_detection.detect([[(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]])[0]
        triangle_ok = Triangle(Point(1, 1), Point(2, 1), Point(1, 2))
        triangle_hole = Triangle(Point(5, 5), Point(6, 5), Point(5, 6))

        result = self.obstacles.filter_triangles_by_holes([triangle_ok, triangle_hole], [hole])
        self.assertEqual(result, [triangle_ok])

    def test_filters_triangles_crossing_hole_boundary(self) -> None:
        hole = self.boundary_detection.detect([[(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]])[0]
        triangle_ok = Triangle(Point(1, 1), Point(2, 1), Point(1, 2))
        triangle_crossing = Triangle(Point(2, 6), Point(10, 6), Point(2, 10))

        result = self.obstacles.filter_triangles_by_holes([triangle_ok, triangle_crossing], [hole])

        self.assertEqual(result, [triangle_ok])

    def test_filters_triangles_covering_hole_vertex(self) -> None:
        hole = self.boundary_detection.detect([[(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]])[0]
        triangle_ok = Triangle(Point(1, 1), Point(2, 1), Point(1, 2))
        triangle_covering_hole = Triangle(Point(0, 0), Point(12, 0), Point(0, 12))

        result = self.obstacles.filter_triangles_by_holes([triangle_ok, triangle_covering_hole], [hole])

        self.assertEqual(result, [triangle_ok])

    def test_normalize_holes_merges_overlapping_regions(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (24, 0), (24, 24), (0, 24), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(4, 4), (12, 4), (12, 12), (4, 12), (4, 4)],
                [(9, 9), (17, 9), (17, 17), (9, 17), (9, 9)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)
        self.assertGreater(len(normalized[0].points), 4)

    def test_normalize_holes_merges_touching_regions(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (24, 0), (24, 24), (0, 24), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(4, 4), (10, 4), (10, 10), (4, 10), (4, 4)],
                [(10, 4), (16, 4), (16, 10), (10, 10), (10, 4)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)

    def test_normalize_holes_keeps_separate_regions_separate(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)],
                [(18, 18), (24, 18), (24, 24), (18, 24), (18, 18)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 2)

    def test_normalize_holes_skips_raster_union_for_separate_regions(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)],
                [(18, 4), (22, 4), (22, 8), (18, 8), (18, 4)],
            ]
        )

        with patch.object(self.obstacles, "_rasterize_holes") as rasterize:
            normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 2)
        rasterize.assert_not_called()

    def test_normalize_holes_uses_raster_union_for_overlapping_regions(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (24, 0), (24, 24), (0, 24), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(4, 4), (12, 4), (12, 12), (4, 12), (4, 4)],
                [(9, 9), (17, 9), (17, 17), (9, 17), (9, 9)],
            ]
        )

        with patch.object(self.obstacles, "_rasterize_holes", wraps=self.obstacles._rasterize_holes) as rasterize:
            normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)
        self.assertGreaterEqual(rasterize.call_count, 1)

    def test_normalize_holes_filters_micro_holes_before_processing(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)],
                [(60, 60), (61, 60), (61, 61), (60, 61), (60, 60)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)

    def test_normalize_holes_merges_transitively_touching_chain(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (80, 0), (80, 40), (0, 40), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(8, 8), (16, 8), (16, 16), (8, 16), (8, 8)],
                [(16, 8), (24, 8), (24, 16), (16, 16), (16, 8)],
                [(24, 8), (32, 8), (32, 16), (24, 16), (24, 8)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)

    def test_normalize_holes_merges_corner_touching_regions(self) -> None:
        shell = self.boundary_detection.detect([[(0, 0), (60, 0), (60, 60), (0, 60), (0, 0)]])[0]
        holes = self.boundary_detection.detect(
            [
                [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
                [(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)],
            ]
        )

        normalized = self.obstacles.normalize_holes(shell, holes)

        self.assertEqual(len(normalized), 1)

    def test_normalize_holes_randomized_cluster_count_regression(self) -> None:
        if not _HAS_SHAPELY:
            self.skipTest("Shapely is required for this randomized regression test.")

        from shapely.ops import unary_union

        rng = random.Random(1337)
        shell = self.boundary_detection.detect([[(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)]])[0]
        shell_poly = Polygon([(p.x, p.y) for p in shell.points])

        for _ in range(40):
            rectangles: list[list[tuple[float, float]]] = []
            count = rng.randint(4, 9)
            for _ in range(count):
                x = rng.randint(10, 170)
                y = rng.randint(10, 170)
                w = rng.randint(8, 20)
                h = rng.randint(8, 20)
                rectangles.append(
                    [
                        (float(x), float(y)),
                        (float(x + w), float(y)),
                        (float(x + w), float(y + h)),
                        (float(x), float(y + h)),
                        (float(x), float(y)),
                    ]
                )

            holes = self.boundary_detection.detect(rectangles)
            if len(holes) < 2:
                continue

            filtered_holes = self.obstacles._filter_micro_holes(shell, holes)  # noqa: SLF001
            if len(filtered_holes) < 2:
                continue

            merge_tol = self.obstacles._merge_tolerance(shell.points, filtered_holes)  # noqa: SLF001
            raw = [Polygon([(p.x, p.y) for p in h.points]).buffer(0) for h in filtered_holes]
            expected_geom = unary_union(raw)
            if merge_tol > 0.0:
                expected_geom = expected_geom.buffer(merge_tol, join_style=2).buffer(-merge_tol, join_style=2).buffer(0)
            expected_geom = expected_geom.intersection(shell_poly).buffer(0)
            expected_clusters = len(self.obstacles._extract_polygons(expected_geom))  # noqa: SLF001
            normalized = self.obstacles.normalize_holes(shell, holes)

            self.assertEqual(
                len(normalized),
                expected_clusters,
                msg=f"Expected {expected_clusters} merged clusters, got {len(normalized)}",
            )


if __name__ == "__main__":
    unittest.main()
