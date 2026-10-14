"""Зауваження 1 для розрізів: вузол розрізу — по номеру на кожен сектор (берег), двійники в моделі й експорті;
імпортовані з JSON розрізи/контури не спрощуються."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

import test_mesh_conformity_invariants as inv
from src.application.services.mesh_postprocessing import split_cut_nodes
from src.domain.entities.mesh import MeshSourceContours
from src.domain.entities.node_type import NodeType
from src.domain.entities.point import Point
from src.infrastructure.export.mesh_result_exporter import export_nodes
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter


def _hexagon_fan() -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Вузол 1 у центрі, 2..7 — шестикутник навколо (повне віяло)."""
    nodes = [(0.0, 0.0)] + [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]
    triangles = [(1, 2 + k, 2 + (k + 1) % 6) for k in range(6)]
    return nodes, triangles


def _sectors(triangles, group) -> list[set[int]]:
    return [{i for i, t in enumerate(triangles) if node in t} for node in group]


def _group_of(groups, node_id: int) -> tuple[int, ...] | None:
    # Кінці розрізу на краю синтетичного віяла теж розщеплюються (там віяло відкрите, як на межі).
    return next((g for g in groups if node_id in g), None)


class TestSplitCutNodes(unittest.TestCase):
    def test_interior_cut_node_gets_two_numbers(self) -> None:
        nodes, triangles = _hexagon_fan()
        new_nodes, new_triangles, groups = split_cut_nodes(nodes, triangles, {frozenset((1, 2)), frozenset((1, 5))})
        group = _group_of(groups, 1)
        self.assertEqual(len(group), 2)
        self.assertEqual(new_nodes[group[1] - 1], nodes[0])
        self.assertEqual(sorted(map(len, _sectors(new_triangles, group))), [3, 3])

    def test_cut_tip_keeps_single_number(self) -> None:
        nodes, triangles = _hexagon_fan()
        _, new_triangles, groups = split_cut_nodes(nodes, triangles, {frozenset((1, 2))})
        self.assertIsNone(_group_of(groups, 1))
        self.assertEqual(sum(1 in t for t in new_triangles), 6)

    def test_junction_of_three_cuts_gets_number_per_sector(self) -> None:
        nodes, triangles = _hexagon_fan()
        cut = {frozenset((1, 2)), frozenset((1, 4)), frozenset((1, 6))}
        _, new_triangles, groups = split_cut_nodes(nodes, triangles, cut)
        group = _group_of(groups, 1)
        self.assertEqual(len(group), 3)
        self.assertEqual(sorted(map(len, _sectors(new_triangles, group))), [2, 2, 2])

    def test_cut_end_on_outer_boundary_gets_two_numbers(self) -> None:
        # Половина віяла: вузол 1 на межі, розріз 1-4 йде всередину.
        nodes, triangles = _hexagon_fan()
        half = triangles[:3]
        _, new_triangles, groups = split_cut_nodes(nodes, half, {frozenset((1, 3))})
        self.assertEqual(sorted(map(len, _sectors(new_triangles, _group_of(groups, 1)))), [1, 2])


class TestCutTwinsInMeshAndExport(unittest.TestCase):
    def test_twins_share_coordinates_types_and_export(self) -> None:
        for name in ("fig22_like.json", inv.HANDDRAWN_FROM_BOUNDARY):
            with self.subTest(case=name):
                case = inv.mesh_case(name)
                idx = case.mesh.indexed_mesh()
                self.assertTrue(idx.twin_groups)
                cut_points = {inv.pkey((p.x, p.y)) for e in case.mesh.cut_edges for p in e}
                for group in idx.twin_groups:
                    coords = {inv.pkey(idx.nodes[n - idx.index_base]) for n in group}
                    self.assertEqual(len(coords), 1)
                    self.assertTrue(coords <= cut_points, "двійник зсунувся з розрізу (згладжування)")
                    self.assertTrue(all(idx.node_type(n) is NodeType.INTERFACE for n in group))
                with tempfile.TemporaryDirectory() as tmp:
                    export_nodes(idx, Path(tmp) / "nodes.json")
                    exported = json.loads((Path(tmp) / "nodes.json").read_text(encoding="utf-8"))["nodes"]
                twins = idx.twin_map()
                for node in exported:
                    self.assertEqual(node["twins"], list(twins.get(node["node_number"], ())))
                group = idx.twin_groups[0]
                record = exported[group[0] - idx.index_base]
                self.assertEqual(record["attribute"], "interface")  # мітка експорту для NodeType.INTERFACE
                self.assertEqual(sorted([record["node_number"], *record["twins"]]), list(group))

    def test_cut_end_on_boundary_is_boundary_and_interface(self) -> None:
        shell = [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)]
        cut = [(0.0, 150.0), (150.0, 150.0)]
        source = MeshSourceContours(outer_boundary=shell, inclusions=[], cuts=[cut])
        mesh, _ = TriangulationAdapter().run_from_contours(
            [[*shell, shell[0]], cut], mode=TriangulationMode.FAST, source_contours=source
        )
        idx = mesh.indexed_mesh()
        groups = {inv.pkey(idx.nodes[g[0] - idx.index_base]): g for g in idx.twin_groups}
        self.assertEqual(len(groups[(0.0, 150.0)]), 2)
        self.assertNotIn((150.0, 150.0), groups, "вістря всередині області має 1 номер")
        for n in groups[(0.0, 150.0)]:
            self.assertIn(n, idx.boundary_nodes)
            self.assertIs(idx.node_type(n), NodeType.INTERFACE)

    def test_reclassifying_by_points_keeps_twins(self) -> None:
        mesh = inv.mesh_case("square_cut.json").mesh
        base = mesh.indexed_mesh()
        cut_points = [p for e in mesh.cut_edges for p in e]
        rebuilt = mesh.indexed_mesh(boundary_points=[], interface_points=cut_points)
        self.assertEqual(len(rebuilt.nodes), len(base.nodes))
        self.assertEqual(rebuilt.triangles, base.triangles)
        self.assertEqual(rebuilt.twin_groups, base.twin_groups)
        self.assertTrue(all(n in rebuilt.interface_nodes for g in base.twin_groups for n in g))
        numbers = {k for t in mesh.linear_triangles() for k in t.node_numbers}
        self.assertEqual(numbers, set(range(1, len(base.nodes) + 1)))


class TestImportedGeometryNotSimplified(unittest.TestCase):
    SHELL = [(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)]
    # (150, 151) лише на 1 px від прямої (100,150)-(200,150): епсилон-спрощення штриха її викидає.
    CUT = [(100, 150), (125, 150), (150, 151), (175, 150), (200, 150), (200, 220)]

    def _cut_vertices(self, source: MeshSourceContours | None) -> set[tuple[float, float]]:
        mesh, _ = TriangulationAdapter().run_from_contours(
            [list(self.SHELL), list(self.CUT)], mode=TriangulationMode.FAST, source_contours=source
        )
        return {inv.pkey((p.x, p.y)) for e in mesh.cut_edges for p in e}

    def test_imported_cut_keeps_every_vertex(self) -> None:
        source = MeshSourceContours(
            outer_boundary=[(float(x), float(y)) for x, y in self.SHELL[:-1]],
            inclusions=[],
            cuts=[[(float(x), float(y)) for x, y in self.CUT]],
        )
        self.assertIn((150.0, 151.0), self._cut_vertices(source))

    def test_drawn_cut_is_still_simplified(self) -> None:
        self.assertNotIn((150.0, 151.0), self._cut_vertices(None))

    def test_imported_boundary_keeps_every_vertex(self) -> None:
        shell = [(0, 0), (150, 1), (300, 0), (300, 300), (0, 300), (0, 0)]
        source = MeshSourceContours(
            outer_boundary=[(float(x), float(y)) for x, y in shell[:-1]], inclusions=[], cuts=[]
        )
        mesh, _ = TriangulationAdapter().run_from_contours([shell], mode=TriangulationMode.FAST, source_contours=source)
        vertices = {inv.pkey((p.x, p.y)) for t in mesh.triangles for p in (t.a, t.b, t.c)}
        self.assertIn((150.0, 1.0), vertices)


if __name__ == "__main__":
    unittest.main()
