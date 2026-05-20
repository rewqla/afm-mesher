import unittest

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

    def test_filters_triangles_inside_hole(self) -> None:
        hole = self.boundary_detection.detect([[(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]])[0]
        triangle_ok = Triangle(Point(1, 1), Point(2, 1), Point(1, 2))
        triangle_hole = Triangle(Point(5, 5), Point(6, 5), Point(5, 6))

        result = self.obstacles.filter_triangles_by_holes([triangle_ok, triangle_hole], [hole])
        self.assertEqual(result, [triangle_ok])


if __name__ == "__main__":
    unittest.main()

