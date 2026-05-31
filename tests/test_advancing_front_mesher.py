import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point
from src.domain.entities.triangle import Triangle
from src.domain.geometry.geometry_utils import distance, orientation, point_in_polygon, segments_intersect


class TestAdvancingFrontMesher(unittest.TestCase):
    def setUp(self) -> None:
        self.mesher = AdvancingFrontMesher(min_triangle_quality=0.01, smoothing_iterations=6)

    def test_subdivide_boundary_uniform(self) -> None:
        square = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ]
        subdivided = self.mesher.subdivide_boundary(square, h=2.0)
        self.assertEqual(len(subdivided), 20)

    def test_generate_mesh_for_square_boundary(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
        ]

        mesh = self.mesher.generate(boundary)
        self.assertGreaterEqual(len(mesh.triangles), 2)
        self._assert_triangles_inside_polygon(mesh, boundary)

    def test_generate_mesh_for_concave_l_shape_boundary(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(8.0, 0.0),
            Point(8.0, 3.0),
            Point(3.0, 3.0),
            Point(3.0, 8.0),
            Point(0.0, 8.0),
        ]

        mesh = self.mesher.generate(boundary)
        self.assertGreaterEqual(len(mesh.triangles), 3)
        self._assert_triangles_inside_polygon(mesh, boundary)

    def test_generate_mesh_with_hole_starts_front_from_hole_boundary(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=50.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        hole = [
            Point(40.0, 40.0),
            Point(60.0, 40.0),
            Point(60.0, 60.0),
            Point(40.0, 60.0),
        ]

        mesh = mesher.generate_with_holes(boundary, [hole])

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        self._assert_triangles_outside_hole(mesh, hole)
        self._assert_hole_boundary_is_used_by_mesh(mesh, hole)

    def test_generate_mesh_with_multiple_holes_starts_front_from_every_hole(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=60.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(160.0, 0.0),
            Point(160.0, 100.0),
            Point(0.0, 100.0),
        ]
        holes = [
            [
                Point(25.0, 25.0),
                Point(45.0, 25.0),
                Point(45.0, 45.0),
                Point(25.0, 45.0),
            ],
            [
                Point(70.0, 45.0),
                Point(95.0, 45.0),
                Point(95.0, 70.0),
                Point(70.0, 70.0),
            ],
            [
                Point(120.0, 20.0),
                Point(140.0, 20.0),
                Point(140.0, 40.0),
                Point(120.0, 40.0),
            ],
        ]

        mesh = mesher.generate_with_holes(boundary, holes)

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        for hole in holes:
            self._assert_triangles_outside_hole(mesh, hole)
            self._assert_hole_boundary_is_used_by_mesh(mesh, hole)

    def test_prepare_cuts_returns_constrained_segments(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        cut = [Point(30.0, 60.0), Point(90.0, 60.0)]
        prepared_boundary = self.mesher._prepare_boundary(boundary)

        segments = self.mesher._prepare_cuts([cut], prepared_boundary, holes=[], tolerance=1e-6)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], cut[0])
        self.assertEqual(segments[0][1], cut[1])

    def test_generate_with_holes_and_cuts_accepts_open_line_as_internal_barrier(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=50.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        cut = [Point(20.0, 50.0), Point(80.0, 50.0)]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[cut])

        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        self.assertTrue(self._mesh_respects_cut_line(mesh, cut))

    def test_front_seed_coverage_includes_boundary_holes_and_bidirectional_cuts(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        holes = [
            [Point(20.0, 20.0), Point(40.0, 20.0), Point(40.0, 40.0), Point(20.0, 40.0)],
            [Point(70.0, 70.0), Point(95.0, 70.0), Point(95.0, 95.0), Point(70.0, 95.0)],
        ]
        cuts = [
            [Point(10.0, 60.0), Point(110.0, 60.0)],
            [Point(60.0, 10.0), Point(60.0, 110.0)],
        ]

        polygon = self.mesher._prepare_boundary(boundary)
        hole_polygons = [self.mesher._prepare_hole_boundary(h) for h in holes]
        cut_segments = self.mesher._prepare_cuts(cuts, polygon, hole_polygons, tolerance=1e-6)
        target_h = self.mesher._resolve_target_step(polygon)

        polygon_sub = self.mesher._subdivide_boundary(polygon, target_h)
        holes_sub = [self.mesher._subdivide_boundary(hole, target_h) for hole in hole_polygons]
        cut_segments_sub = self.mesher._subdivide_cut_segments(cut_segments, target_h)

        front = self.mesher._build_initial_front(polygon_sub)
        for hole in holes_sub:
            front.extend(self.mesher._build_initial_front(hole))
        for a, b in cut_segments_sub:
            front.append((a, b))
            front.append((b, a))

        front_keys = {self._directed_edge_key(edge) for edge in front}
        for edge in self.mesher._build_initial_front(polygon_sub):
            self.assertIn(self._directed_edge_key(edge), front_keys)
        for hole in holes_sub:
            for edge in self.mesher._build_initial_front(hole):
                self.assertIn(self._directed_edge_key(edge), front_keys)
        for edge in cut_segments_sub:
            self.assertIn(self._directed_edge_key(edge), front_keys)
            self.assertIn(self._directed_edge_key((edge[1], edge[0])), front_keys)

    def test_no_bridge_across_long_straight_cut(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=12.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 100.0),
            Point(0.0, 100.0),
        ]
        cut = [Point(10.0, 50.0), Point(110.0, 50.0)]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[cut])

        self.assertGreater(len(mesh.triangles), 0)
        cut_segment = (cut[0], cut[1])
        eps = 1e-6
        for triangle in mesh.triangles:
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)):
                # Shared constrained edge nodes are allowed.
                if self._same_undirected_edge(edge, cut_segment):
                    continue
                side_a = edge[0].y - 50.0
                side_b = edge[1].y - 50.0
                crosses_sides = (side_a > eps and side_b < -eps) or (side_a < -eps and side_b > eps)
                if not crosses_sides:
                    continue
                intersects_cut_interior = segments_intersect(
                    edge[0],
                    edge[1],
                    cut_segment[0],
                    cut_segment[1],
                    include_endpoints=False,
                )
                self.assertFalse(intersects_cut_interior)

    def test_hole_hugging_stability_with_small_hole_and_coarse_target_h(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=30.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        hole = [
            Point(56.0, 56.0),
            Point(64.0, 56.0),
            Point(64.0, 64.0),
            Point(56.0, 64.0),
        ]

        mesh = mesher.generate_with_holes(boundary, [hole])
        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_outside_hole(mesh, hole)

        cx = sum(p.x for p in hole) / len(hole)
        cy = sum(p.y for p in hole) / len(hole)
        ring_min = 8.0
        ring_max = 26.0
        ring_triangles = 0
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            dx = centroid.x - cx
            dy = centroid.y - cy
            r2 = dx * dx + dy * dy
            if ring_min * ring_min <= r2 <= ring_max * ring_max:
                ring_triangles += 1

        self.assertGreaterEqual(ring_triangles, 4)

    def test_intersecting_cuts_split_topology_and_segments_exist_in_final_mesh(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=20.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        cut_a = [Point(10.0, 50.0), Point(90.0, 50.0)]
        cut_b = [Point(50.0, 10.0), Point(50.0, 90.0)]

        prepared_boundary = mesher._prepare_boundary(boundary)
        prepared_segments = mesher._prepare_cuts([cut_a, cut_b], prepared_boundary, holes=[], tolerance=1e-6)

        intersection = Point(50.0, 50.0)
        expected_split_segments = [
            (Point(10.0, 50.0), intersection),
            (intersection, Point(90.0, 50.0)),
            (Point(50.0, 10.0), intersection),
            (intersection, Point(50.0, 90.0)),
        ]
        for segment in expected_split_segments:
            self.assertIn(segment, prepared_segments)

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[cut_a, cut_b])
        mesh_edges = {
            self._edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }

        target_h = mesher._resolve_target_step(prepared_boundary)
        subdivided_expected = mesher._subdivide_cut_segments(expected_split_segments, target_h)
        self.assertGreater(len(subdivided_expected), 0)
        for segment in subdivided_expected:
            self.assertIn(self._edge_key(segment), mesh_edges)

    def test_cut_endpoint_on_boundary_vertex_is_stable_and_preserved(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=15.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        cut = [Point(0.0, 0.0), Point(70.0, 50.0)]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[cut])
        self.assertGreater(len(mesh.triangles), 0)

        polygon = mesher._prepare_boundary(boundary)
        cut_segments = mesher._prepare_cuts([cut], polygon, holes=[], tolerance=1e-6)
        target_h = mesher._resolve_target_step(polygon)
        cut_segments_sub = mesher._subdivide_cut_segments(cut_segments, target_h)
        front = mesher._build_initial_front(mesher._subdivide_boundary(polygon, target_h))
        for a, b in cut_segments_sub:
            front.append((a, b))
            front.append((b, a))

        # No proper self-intersections in seeded front (shared endpoints and reverse duplicates are allowed).
        for i, first in enumerate(front):
            for second in front[i + 1:]:
                if self._same_undirected_edge(first, second):
                    continue
                if first[0] == second[0] or first[0] == second[1] or first[1] == second[0] or first[1] == second[1]:
                    continue
                self.assertFalse(
                    segments_intersect(first[0], first[1], second[0], second[1], include_endpoints=False)
                )

        mesh_edges = {
            self._edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }
        for segment in cut_segments_sub:
            self.assertIn(self._edge_key(segment), mesh_edges)

    def test_cut_endpoint_near_boundary_vertex_snaps_and_no_short_duplicate_segments(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        polygon = self.mesher._prepare_boundary(boundary)
        tolerance = 1e-3
        cut = [Point(5e-4, 5e-4), Point(0.0, 0.0), Point(60.0, 40.0)]

        segments = self.mesher._prepare_cuts([cut], polygon, holes=[], tolerance=tolerance)

        self.assertGreaterEqual(len(segments), 1)
        self.assertEqual(segments[0][0], Point(0.0, 0.0))
        for a, b in segments:
            self.assertGreater(distance(a, b), tolerance)

    def test_find_advancement_uses_reversed_edge_recovery_when_forward_fails(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(40.0, 0.0),
            Point(40.0, 40.0),
            Point(0.0, 40.0),
        ]
        polygon = self.mesher._prepare_boundary(boundary)
        front = [(Point(10.0, 10.0), Point(20.0, 10.0))]
        recovery_candidate = Point(15.0, 20.0)

        def fake_find_best_node(
            a: Point,
            b: Point,
            _polygon: list[Point],
            _holes: list[list[Point]],
            _cut_segments: list[tuple[Point, Point]],
            _front: list[tuple[Point, Point]],
            _target_step: float,
            *_args: object,
        ) -> Point | None:
            if a == front[0][1] and b == front[0][0]:
                return recovery_candidate
            return None

        with patch.object(self.mesher, "_find_best_node", side_effect=fake_find_best_node):
            advancement = self.mesher._find_advancement(
                front=front,
                polygon=polygon,
                holes=[],
                cut_segments=[],
                target_step=10.0,
            )

        self.assertIsNotNone(advancement)
        idx, a, b, candidate = advancement  # type: ignore[misc]
        self.assertEqual(idx, 0)
        self.assertEqual((a, b), (front[0][1], front[0][0]))
        self.assertEqual(candidate, recovery_candidate)

    def test_find_advancement_returns_none_when_forward_and_reversed_fail(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(40.0, 0.0),
            Point(40.0, 40.0),
            Point(0.0, 40.0),
        ]
        polygon = self.mesher._prepare_boundary(boundary)
        front = [(Point(10.0, 10.0), Point(20.0, 10.0))]

        with patch.object(self.mesher, "_find_best_node", return_value=None):
            advancement = self.mesher._find_advancement(
                front=front,
                polygon=polygon,
                holes=[],
                cut_segments=[],
                target_step=10.0,
            )

        self.assertIsNone(advancement)

    def test_reject_invalid_cut_through_hole_with_clear_error(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        hole = [Point(50.0, 50.0), Point(70.0, 50.0), Point(70.0, 70.0), Point(50.0, 70.0)]
        cut = [Point(20.0, 60.0), Point(100.0, 60.0)]  # passes through hole

        polygon = self.mesher._prepare_boundary(boundary)
        hole_polygon = self.mesher._prepare_hole_boundary(hole)

        with self.assertRaisesRegex(ValueError, "hole"):
            self.mesher._prepare_cuts([cut], polygon, holes=[hole_polygon], tolerance=1e-6)

    def test_short_noise_cut_is_rejected_by_filtering_policy(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        polygon = self.mesher._prepare_boundary(boundary)
        tolerance = 1e-3
        noise_cut = [Point(25.0, 25.0), Point(25.0 + 1e-6, 25.0 + 1e-6)]

        segments = self.mesher._prepare_cuts([noise_cut], polygon, holes=[], tolerance=tolerance)

        self.assertEqual(segments, [])

    def test_multiple_holes_and_multiple_cuts_regression(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=16.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(140.0, 0.0),
            Point(140.0, 120.0),
            Point(0.0, 120.0),
        ]
        holes = [
            [Point(20.0, 20.0), Point(40.0, 20.0), Point(40.0, 40.0), Point(20.0, 40.0)],
            [Point(90.0, 20.0), Point(115.0, 20.0), Point(115.0, 45.0), Point(90.0, 45.0)],
            [Point(60.0, 70.0), Point(82.0, 70.0), Point(82.0, 92.0), Point(60.0, 92.0)],
        ]
        cuts = [
            [Point(10.0, 60.0), Point(130.0, 60.0)],
            [Point(55.0, 10.0), Point(55.0, 110.0)],
        ]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=holes, cuts=cuts)
        self.assertGreater(len(mesh.triangles), 0)
        self._assert_triangles_inside_polygon(mesh, boundary)
        for hole in holes:
            self._assert_triangles_outside_hole(mesh, hole)

        polygon = mesher._prepare_boundary(boundary)
        prepared_holes = [mesher._prepare_hole_boundary(hole) for hole in holes]
        cut_segments = mesher._prepare_cuts(cuts, polygon, prepared_holes, tolerance=1e-6)
        cut_segments_sub = mesher._subdivide_cut_segments(cut_segments, mesher._resolve_target_step(polygon))

        mesh_edges = {
            self._edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }
        for segment in cut_segments_sub:
            self.assertIn(self._edge_key(segment), mesh_edges)

        self._assert_mesh_edges_do_not_cross_cut_segments(mesh, cut_segments_sub)

    def test_fallback_safety_with_forced_fallback(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            max_iterations_factor=1,
            target_edge_length=20.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(120.0, 0.0),
            Point(120.0, 120.0),
            Point(0.0, 120.0),
        ]
        holes = [[Point(48.0, 48.0), Point(72.0, 48.0), Point(72.0, 72.0), Point(48.0, 72.0)]]
        cuts = [[Point(10.0, 35.0), Point(110.0, 35.0)]]

        polygon = mesher._prepare_boundary(boundary)
        prepared_holes = [mesher._prepare_hole_boundary(hole) for hole in holes]
        cut_segments = mesher._prepare_cuts(cuts, polygon, prepared_holes, tolerance=1e-6)

        # Build a single-loop front for deterministic fallback check.
        front = mesher._build_initial_front(mesher._subdivide_boundary(polygon, mesher._resolve_target_step(polygon)))
        with patch.object(mesher, "_find_advancement", return_value=None):
            fallback_triangles = mesher._fallback_triangulate_front(front, polygon, prepared_holes, cut_segments)

        self.assertGreater(len(fallback_triangles), 0)
        fallback_mesh = Mesh(triangles=fallback_triangles)
        self._assert_triangles_inside_polygon(fallback_mesh, boundary)
        for hole in holes:
            self._assert_triangles_outside_hole(fallback_mesh, hole)
        self._assert_mesh_edges_do_not_cross_cut_segments(fallback_mesh, cut_segments)

        # No long "spikes": cap longest edge by boundary diagonal in this constrained fallback.
        max_edge = 0.0
        for triangle in fallback_mesh.triangles:
            max_edge = max(
                max_edge,
                distance(triangle.a, triangle.b),
                distance(triangle.b, triangle.c),
                distance(triangle.c, triangle.a),
            )
        self.assertLessEqual(max_edge, 180.0)

    def test_triangle_size_near_long_cut_is_bounded_by_target_h_factor(self) -> None:
        target_h = 12.0
        k = 2.6
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=target_h,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(140.0, 0.0),
            Point(140.0, 120.0),
            Point(0.0, 120.0),
        ]
        cuts = [[Point(10.0, 60.0), Point(130.0, 60.0)]]

        mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=cuts)
        self.assertGreater(len(mesh.triangles), 0)

        polygon = mesher._prepare_boundary(boundary)
        cut_segments = mesher._prepare_cuts(cuts, polygon, holes=[], tolerance=1e-6)
        cut_segments_sub = mesher._subdivide_cut_segments(cut_segments, target_h)
        cut_vertices = {p for seg in cut_segments_sub for p in seg}

        max_incident_edge = 0.0
        has_incident = False
        for triangle in mesh.triangles:
            tri_edges = ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
            if not any(edge[0] in cut_vertices or edge[1] in cut_vertices for edge in tri_edges):
                continue
            has_incident = True
            for a, b in tri_edges:
                max_incident_edge = max(max_incident_edge, distance(a, b))

        self.assertTrue(has_incident)
        self.assertLessEqual(max_incident_edge, k * target_h)

    def test_orientation_invariants_after_generation_and_smoothing(self) -> None:
        target_h = 14.0
        mesher_no_smooth = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=target_h,
            smoothing_iterations=0,
        )
        mesher_smooth = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=target_h,
            smoothing_iterations=6,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(130.0, 0.0),
            Point(130.0, 110.0),
            Point(0.0, 110.0),
        ]
        holes = [
            [Point(25.0, 25.0), Point(45.0, 25.0), Point(45.0, 45.0), Point(25.0, 45.0)],
            [Point(80.0, 60.0), Point(100.0, 60.0), Point(100.0, 82.0), Point(80.0, 82.0)],
        ]
        cuts = [
            [Point(10.0, 55.0), Point(120.0, 55.0)],
            [Point(60.0, 10.0), Point(60.0, 100.0)],
        ]

        mesh_raw = mesher_no_smooth.generate_with_holes_and_cuts(boundary, holes=holes, cuts=cuts)
        mesh_smooth = mesher_smooth.generate_with_holes_and_cuts(boundary, holes=holes, cuts=cuts)

        self.assertGreater(len(mesh_raw.triangles), 0)
        self.assertGreater(len(mesh_smooth.triangles), 0)
        for triangle in mesh_raw.triangles:
            self.assertGreater(orientation(triangle.a, triangle.b, triangle.c), 0.0)
        for triangle in mesh_smooth.triangles:
            self.assertGreater(orientation(triangle.a, triangle.b, triangle.c), 0.0)

    def test_subdivide_cut_segments_splits_long_segment_by_target_h(self) -> None:
        segment = (Point(0.0, 0.0), Point(100.0, 0.0))
        subdivided = self.mesher._subdivide_cut_segments([segment], h=20.0)

        self.assertEqual(len(subdivided), 5)
        self.assertEqual(subdivided[0][0], Point(0.0, 0.0))
        self.assertEqual(subdivided[-1][1], Point(100.0, 0.0))
        for a, b in subdivided:
            self.assertLessEqual(abs((b.x - a.x) ** 2 + (b.y - a.y) ** 2 - 400.0), 1e-6)

    def test_prepare_cuts_splits_intersecting_cuts(self) -> None:
        boundary = [
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
        prepared_boundary = self.mesher._prepare_boundary(boundary)
        cut_a = [Point(10.0, 50.0), Point(90.0, 50.0)]
        cut_b = [Point(50.0, 10.0), Point(50.0, 90.0)]

        segments = self.mesher._prepare_cuts([cut_a, cut_b], prepared_boundary, holes=[], tolerance=1e-6)

        intersection = Point(50.0, 50.0)
        self.assertIn((Point(10.0, 50.0), intersection), segments)
        self.assertIn((intersection, Point(90.0, 50.0)), segments)
        self.assertIn((Point(50.0, 10.0), intersection), segments)
        self.assertIn((intersection, Point(50.0, 90.0)), segments)

    def test_triangle_respects_cut_segments_rejects_crossing(self) -> None:
        cut_segments = [(Point(0.0, 0.0), Point(10.0, 0.0))]
        # Triangle edge (5,-1)->(5,1) crosses the cut segment.
        allowed = self.mesher._triangle_respects_cut_segments(
            Point(5.0, -1.0),
            Point(5.0, 1.0),
            Point(8.0, 1.0),
            cut_segments,
        )
        self.assertFalse(allowed)

    def test_triangle_respects_cut_segments_accepts_shared_constrained_edge(self) -> None:
        cut_segments = [(Point(0.0, 0.0), Point(10.0, 0.0))]
        # Triangle uses constrained segment as one edge; this must be allowed.
        allowed = self.mesher._triangle_respects_cut_segments(
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(5.0, 4.0),
            cut_segments,
        )
        self.assertTrue(allowed)

    def test_rejects_degenerate_boundary(self) -> None:
        boundary = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
        with self.assertRaises(ValueError):
            self.mesher.generate(boundary)

    def test_build_topology_uses_global_linear_triangle_node_numbers(self) -> None:
        mesh = Mesh(
            triangles=[
                Triangle(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
                Triangle(Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
            ]
        )
        boundary = [Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)]

        positions, triangles, _, _, _ = self.mesher._build_topology(mesh, boundary)

        self.assertEqual(sorted(positions.keys()), [1, 2, 3, 4])
        self.assertEqual(triangles[0][1], triangles[1][0])
        self.assertEqual(triangles[0][2], triangles[1][2])

    def test_generate_with_holes_and_cuts_retries_after_single_pass_failure(self) -> None:
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.0,
            target_edge_length=20.0,
            smoothing_iterations=0,
        )
        boundary = [
            Point(0.0, 0.0),
            Point(80.0, 0.0),
            Point(80.0, 80.0),
            Point(0.0, 80.0),
        ]
        fake_mesh = Mesh(triangles=[Triangle(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0))])

        with (
            patch.object(
                mesher,
                "_generate_single_pass",
                side_effect=[ValueError("AFM stalled: active front cannot be advanced further."), (fake_mesh, boundary, [])],
            ) as generate_single_pass,
            patch.object(mesher, "_validate_cut_segments_are_mesh_edges", return_value=None),
        ):
            mesh = mesher.generate_with_holes_and_cuts(boundary, holes=[], cuts=[])

        self.assertEqual(mesh.triangles, fake_mesh.triangles)
        self.assertEqual(generate_single_pass.call_count, 2)

    def _assert_triangles_inside_polygon(self, mesh, polygon: list[Point]) -> None:
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertTrue(point_in_polygon(centroid, polygon, include_boundary=True))

    def _assert_triangles_outside_hole(self, mesh, hole: list[Point]) -> None:
        hole_edges = [(hole[idx], hole[(idx + 1) % len(hole)]) for idx in range(len(hole))]
        for triangle in mesh.triangles:
            centroid = Point(
                (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
                (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            )
            self.assertFalse(point_in_polygon(centroid, hole, include_boundary=False))
            triangle_edges = ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
            for edge in triangle_edges:
                for hole_edge in hole_edges:
                    if self._same_undirected_edge(edge, hole_edge):
                        continue
                    self.assertFalse(
                        segments_intersect(edge[0], edge[1], hole_edge[0], hole_edge[1], include_endpoints=False)
                    )

    def _assert_hole_boundary_is_used_by_mesh(self, mesh, hole: list[Point]) -> None:
        mesh_edges = {
            self._edge_key(edge)
            for triangle in mesh.triangles
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a))
        }
        for idx in range(len(hole)):
            self.assertIn(self._edge_key((hole[idx], hole[(idx + 1) % len(hole)])), mesh_edges)

    def _same_undirected_edge(self, first: tuple[Point, Point], second: tuple[Point, Point]) -> bool:
        return self._edge_key(first) == self._edge_key(second)

    def _edge_key(self, edge: tuple[Point, Point]) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = edge
        return tuple(sorted(((round(a.x, 6), round(a.y, 6)), (round(b.x, 6), round(b.y, 6)))))

    def _directed_edge_key(self, edge: tuple[Point, Point]) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = edge
        return (round(a.x, 6), round(a.y, 6)), (round(b.x, 6), round(b.y, 6))

    def _mesh_respects_cut_line(self, mesh: Mesh, cut: list[Point]) -> bool:
        if len(cut) < 2:
            return True
        line = (cut[0], cut[-1])
        for triangle in mesh.triangles:
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)):
                if segments_intersect(edge[0], edge[1], line[0], line[1], include_endpoints=False):
                    return False
        return True

    def _assert_mesh_edges_do_not_cross_cut_segments(self, mesh: Mesh, cut_segments: list[tuple[Point, Point]]) -> None:
        for triangle in mesh.triangles:
            for edge in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)):
                for cut_segment in cut_segments:
                    if self._same_undirected_edge(edge, cut_segment):
                        continue
                    self.assertFalse(
                        segments_intersect(edge[0], edge[1], cut_segment[0], cut_segment[1], include_endpoints=False)
                    )


if __name__ == "__main__":
    unittest.main()
