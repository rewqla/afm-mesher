import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from src.application.services.boundary_detection_service import BoundaryDetectionService
from src.application.services.obstacle_processor import ObstacleProcessor
from src.application.services.region_classifier import RegionClassifier
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle


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


if __name__ == "__main__":
    unittest.main()
