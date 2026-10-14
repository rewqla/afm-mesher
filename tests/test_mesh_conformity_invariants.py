"""Інваріанти конформності сітки на кейсах з diagnostics/cases (зауваження викладача 1–3).

Тести, що зараз падають, позначені @unittest.expectedFailure з посиланням на крок фіксу;
декоратор знімається в кроці, який виправляє відповідну проблему.

Перевірки геометрії — чистий Python (логіка з diagnostics/diagnose_mesh.py).
Кейс square_cut_offgrid раніше зависав, тому він тріангулюється в дочірньому процесі з таймаутом,
щоб регресія не повісила набір (запуск: `python tests/test_mesh_conformity_invariants.py --child <case>.json <mode>`).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import src.application.services.advancing_front_mesher as afm_module
import src.presentation.tri_debug as tri_debug_module
import src.presentation.triangulation_adapter as adapter_module
from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.mesh import Mesh, MeshSourceContours
from src.infrastructure.export.mesh_result_exporter import export_nodes, export_triangles
from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter, ImportedCoordinates
from src.presentation.tools import TriangulationMode

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "diagnostics" / "cases"
LOG_PATH = Path(tempfile.gettempdir()) / "afm_mesher_conformity_tests.log"
BALANCED = TriangulationMode.BALANCED
FAST = TriangulationMode.FAST
OFFGRID_CASE = "square_cut_offgrid.json"
OFFGRID_TIMEOUT_S = 30.0

KEY = 6
TOL = 1e-6

Pt = tuple[float, float]
Tri = tuple[Pt, Pt, Pt]


# ---------------------------------------------------------------- геометрія
def pkey(p) -> Pt:
    return (round(p[0], KEY), round(p[1], KEY))


def cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def polyline_dist(p, line) -> float:
    return min(seg_dist(p, a, b) for a, b in zip(line[:-1], line[1:]))


def closed(ring) -> list[Pt]:
    return [*ring, ring[0]] if ring[0] != ring[-1] else list(ring)


def ring_area(ring) -> float:
    n = len(ring)
    return sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n)) / 2.0


def point_in_ring(p, ring) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            if x > p[0]:
                inside = not inside
    return inside


def centroid(t) -> Pt:
    return ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0)


def _ccw(t):
    return t if cross(*t) > 0 else (t[0], t[2], t[1])


def _line_inter(p, q, a, b):
    d1, d2 = cross(a, b, p), cross(a, b, q)
    t = d1 / (d1 - d2) if d1 != d2 else 0.0
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def tri_intersection_area(t1, t2) -> float:
    """Площа перетину двох трикутників (Sutherland–Hodgman)."""
    t1 = _ccw(t1)
    out = list(_ccw(t2))
    for i in range(3):
        a, b = t1[i], t1[(i + 1) % 3]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            cin, pin = cross(a, b, cur) >= 0, cross(a, b, prev) >= 0
            if cin:
                if not pin:
                    out.append(_line_inter(prev, cur, a, b))
                out.append(cur)
            elif pin:
                out.append(_line_inter(prev, cur, a, b))
    return abs(ring_area(out)) if len(out) >= 3 else 0.0


def proper_cross(p1, p2, q1, q2, dist_eps: float = 1e-4) -> bool:
    """Строгий перетин відрізків; дотик кінцем (ближче dist_eps) перетином не вважається."""
    lp, lq = math.dist(p1, p2), math.dist(q1, q2)
    if lp == 0 or lq == 0:
        return False
    d1, d2 = cross(p1, p2, q1) / lp, cross(p1, p2, q2) / lp
    d3, d4 = cross(q1, q2, p1) / lq, cross(q1, q2, p2) / lq
    if min(abs(d1), abs(d2), abs(d3), abs(d4)) <= dist_eps:
        return False
    return d1 * d2 < 0 and d3 * d4 < 0


class Grid:
    def __init__(self, cell: float) -> None:
        self.cell = max(cell, 1e-9)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _range(self, pts):
        c = self.cell
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (math.floor(min(xs) / c), math.floor(max(xs) / c), math.floor(min(ys) / c), math.floor(max(ys) / c))

    def insert(self, idx: int, pts) -> None:
        x0, x1, y0, y1 = self._range(pts)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                self.cells[(gx, gy)].append(idx)

    def query(self, pts) -> set[int]:
        x0, x1, y0, y1 = self._range(pts)
        out: set[int] = set()
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                out.update(self.cells.get((gx, gy), ()))
        return out


def tri_edges(t) -> list[tuple[Pt, Pt]]:
    return [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]


# ---------------------------------------------------------------- розрізи
@dataclass(frozen=True, slots=True)
class CutGeometry:
    """Розрізи так, як їх отримав мешер (адаптер може розбити одну ламану на кілька)."""

    polylines: list[list[Pt]]
    interior_tips: frozenset[Pt]

    @classmethod
    def build(cls, cuts: list[list[Pt]], shell: list[Pt]) -> "CutGeometry":
        shell_line = closed(shell)
        ends = Counter(pkey(p) for cut in cuts for p in (cut[0], cut[-1]))
        # Кінець, спільний для двох шматків, — внутрішній вузол розрізу; одиничний кінець — вістря,
        # якщо він не на зовнішній межі.
        tips = frozenset(e for e, n in ends.items() if n == 1 and polyline_dist(e, shell_line) >= TOL)
        return cls(polylines=cuts, interior_tips=tips)

    @property
    def segments(self) -> list[tuple[Pt, Pt]]:
        return [s for cut in self.polylines for s in zip(cut[:-1], cut[1:])]

    def contains(self, p) -> bool:
        return any(polyline_dist(p, cut) < TOL for cut in self.polylines)

    def side(self, p) -> int:
        a, b = min(self.segments, key=lambda s: seg_dist(p, *s))
        return 1 if cross(a, b, p) > 0 else -1


# ---------------------------------------------------------------- спільний хелпер перевірок
def analyze_conformity(tris: list[Tri], shell: list[Pt], holes: list[list[Pt]], cuts: CutGeometry) -> dict:
    """Повертає списки порушень (порожній список = інваріант виконано) і площі."""
    areas = [abs(cross(*t)) / 2.0 for t in tris]
    scale = max(math.dist(a, b) for t in tris for a, b in tri_edges(t))
    grid = Grid(scale)
    for i, t in enumerate(tris):
        grid.insert(i, t)

    overlap_pairs = []
    for i, t in enumerate(tris):
        for j in grid.query(t):
            if j <= i:
                continue
            inter = tri_intersection_area(t, tris[j])
            if inter > 1e-9 and inter > 1e-6 * min(areas[i], areas[j]):
                overlap_pairs.append((centroid(t), centroid(tris[j]), round(inter, 3)))

    edge_tris: dict[frozenset, list[int]] = defaultdict(list)
    incident: dict[Pt, list[int]] = defaultdict(list)
    for i, t in enumerate(tris):
        for p, q in tri_edges(t):
            edge_tris[frozenset((pkey(p), pkey(q)))].append(i)
        for p in t:
            incident[pkey(p)].append(i)
    edges = list(edge_tris)
    edges_over_2 = [tuple(e) for e, ts in edge_tris.items() if len(ts) > 2]

    egrid = Grid(scale)
    for k, e in enumerate(edges):
        egrid.insert(k, list(e))
    vertices = sorted(incident)
    hanging = []
    for v in vertices:
        for k in egrid.query([v]):
            e = edges[k]
            if v in e:
                continue
            p, q = tuple(e)
            length = math.dist(p, q)
            if length == 0:
                continue
            t = ((v[0] - p[0]) * (q[0] - p[0]) + (v[1] - p[1]) * (q[1] - p[1])) / (length * length)
            if 1e-9 < t < 1 - 1e-9 and seg_dist(v, p, q) <= TOL * max(length, 1.0):
                hanging.append((v, tuple(e)))

    rings = [closed(r) for r in (shell, *holes)]

    def on_constraint(v) -> bool:
        return any(polyline_dist(v, r) < TOL for r in rings) or cuts.contains(v)

    valence3 = [v for v in vertices if len(incident[v]) == 3 and not on_constraint(v)]

    cut_node_one_side, cut_edge_unpaired, cut_crossing, cut_uncovered = [], [], [], []
    if cuts.polylines:
        for v in vertices:
            if not cuts.contains(v) or v in cuts.interior_tips:
                continue
            if len({cuts.side(centroid(tris[i])) for i in incident[v]}) < 2:
                cut_node_one_side.append(v)
        covered = 0.0
        for e, ts in edge_tris.items():
            p, q = tuple(e)
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if all(cuts.contains(x) for x in (p, q, mid)):
                covered += math.dist(p, q)
                if sorted(cuts.side(centroid(tris[i])) for i in ts) != [-1, 1]:
                    cut_edge_unpaired.append(mid)
            if any(proper_cross(p, q, a, b) for a, b in cuts.segments):
                cut_crossing.append(mid)
        cut_length = sum(math.dist(a, b) for a, b in cuts.segments)
        if abs(cut_length - covered) > 1e-3:
            cut_uncovered.append(round(cut_length - covered, 3))

    return {
        "domain_area": abs(ring_area(shell)) - sum(abs(ring_area(h)) for h in holes),
        "sum_area": sum(areas),
        "overlap_area": sum(x[2] for x in overlap_pairs),
        "overlap_pairs": overlap_pairs,
        "edges_over_2": edges_over_2,
        "hanging": hanging,
        "valence3": valence3,
        "cut_node_one_side": cut_node_one_side,
        "cut_edge_unpaired": cut_edge_unpaired,
        "cut_crossing": cut_crossing,
        "cut_uncovered": cut_uncovered,
    }


def overlap_violations(report: dict) -> list[str]:
    out = []
    if abs(report["sum_area"] - report["domain_area"]) > 1e-6 * report["domain_area"]:
        out.append(f"сума площ {report['sum_area']:.3f} != площа області {report['domain_area']:.3f}")
    for key in ("overlap_pairs", "edges_over_2"):
        if report[key]:
            out.append(f"{key}: {len(report[key])}, напр. {report[key][:3]}")
    return out


CUT_KEYS = ("cut_node_one_side", "cut_edge_unpaired", "cut_crossing", "cut_uncovered")


def cut_violations(report: dict) -> list[str]:
    return [f"{k}: {len(report[k])}, напр. {report[k][:3]}" for k in CUT_KEYS if report[k]]


# ---------------------------------------------------------------- тріангуляція кейсу
@dataclass(slots=True)
class MeshedCase:
    name: str
    imported: ImportedCoordinates
    mesh: Mesh
    shell: list[Pt]
    holes: list[list[Pt]]
    cuts: CutGeometry

    @property
    def tris(self) -> list[Tri]:
        return [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in self.mesh.triangles]


@lru_cache(maxsize=None)
def mesh_case(name: str, mode: TriangulationMode = BALANCED) -> MeshedCase:
    importer = CoordinateJsonImporter()
    imported = importer.load(str(CASES_DIR / name))
    # fig22_handdrawn* — записані піксельні штрихи пера: в UI вони йдуть шляхом canvas (без
    # source_contours, зі спрощенням). Решта кейсів — точні вершини імпорту (без спрощення).
    source = None if name.startswith("fig22_handdrawn") else MeshSourceContours(
        outer_boundary=list(imported.outer_boundary),
        inclusions=[(list(v), t) for v, t in imported.inclusions],
        cuts=[list(c) for c in imported.cuts],
    )
    captured: dict[str, tuple] = {}

    class RecordingMesher(afm_module.AdvancingFrontMesher):
        def generate_with_holes_and_cuts(self, boundary, holes, cuts, **kwargs):
            mesh = super().generate_with_holes_and_cuts(boundary, holes, cuts, **kwargs)
            captured["input"] = (list(boundary), [list(h) for h in holes], [list(c) for c in cuts])
            return mesh

    with patch.object(adapter_module, "AdvancingFrontMesher", RecordingMesher), \
            patch.object(afm_module, "_AFM_DEBUG_LOG_PATH", LOG_PATH), \
            patch.object(tri_debug_module, "_LOG_PATH", LOG_PATH):
        mesh, _ = adapter_module.TriangulationAdapter().run_from_contours(
            importer.to_canvas_contours(imported), mode=mode, source_contours=source
        )
    boundary, holes, cuts = captured["input"]
    shell = [(p.x, p.y) for p in boundary]
    return MeshedCase(
        name=name,
        imported=imported,
        mesh=mesh,
        shell=shell,
        holes=[[(p.x, p.y) for p in h] for h in holes],
        cuts=CutGeometry.build([[(p.x, p.y) for p in c] for c in cuts], shell),
    )


@lru_cache(maxsize=None)
def conformity(name: str, mode: TriangulationMode = BALANCED) -> dict:
    case = mesh_case(name, mode)
    return analyze_conformity(case.tris, case.shell, case.holes, case.cuts)


# ---------------------------------------------------------------- IndexedMesh: двійники та нумерація
def _node_xy(idx: IndexedMesh, node_id: int) -> Pt:
    return idx.nodes[node_id - idx.index_base]


def _coordinate_groups(idx: IndexedMesh) -> dict[Pt, list[int]]:
    groups: dict[Pt, list[int]] = defaultdict(list)
    for offset, xy in enumerate(idx.nodes):
        groups[pkey(xy)].append(idx.index_base + offset)
    return groups


def _incident_centroids(idx: IndexedMesh) -> dict[int, list[Pt]]:
    out: dict[int, list[Pt]] = defaultdict(list)
    for tri in idx.triangles:
        c = centroid([_node_xy(idx, n) for n in tri])
        for n in tri:
            out[n].append(c)
    return out


def twin_violations(idx: IndexedMesh, on_interface, side, single_id, skip=lambda xy: False) -> list[str]:
    """Кожна точка інтерфейсу (on_interface) має 2 номери, трикутники кожної сторони (side)
    посилаються лише на «свій»; точки single_id та всі інші вузли — рівно 1 номер.
    Точки skip (інший інтерфейс, що перевіряється окремо) пропускаються."""
    groups = _coordinate_groups(idx)
    centroids = _incident_centroids(idx)
    problems = []
    for xy, ids in groups.items():
        if skip(xy):
            continue
        twin = on_interface(xy) and not single_id(xy)
        if len(ids) != (2 if twin else 1):
            problems.append(f"{xy}: номерів {len(ids)}, очікується {2 if twin else 1}")
            continue
        if twin:
            sides = [{side(c) for c in centroids[n]} for n in ids]
            if any(len(s) != 1 for s in sides) or sides[0] == sides[1]:
                problems.append(f"{xy}: номери {ids} змішують береги {sides}")
    return problems


def numbering_violations(mesh: Mesh) -> list[str]:
    """Суцільні ряди 1..N (вузли) та 1..M (трикутники) — в IndexedMesh і в експорті."""
    idx = mesh.indexed_mesh()
    n, problems = len(idx.nodes), []
    if idx.index_base != 1:
        problems.append(f"index_base={idx.index_base}")
    used = {i for t in idx.triangles for i in t}
    if used != set(range(1, n + 1)):
        problems.append(f"вузли без трикутників або поза 1..{n}: {sorted(set(range(1, n + 1)) ^ used)[:10]}")
    with tempfile.TemporaryDirectory() as tmp:
        export_nodes(idx, Path(tmp) / "nodes.json")
        export_triangles(mesh, Path(tmp) / "triangles.json")
        nodes = json.loads((Path(tmp) / "nodes.json").read_text(encoding="utf-8"))["nodes"]
        tris = json.loads((Path(tmp) / "triangles.json").read_text(encoding="utf-8"))["triangles"]
    if [x["node_number"] for x in nodes] != list(range(1, n + 1)):
        problems.append("експорт вузлів: номери не 1..N")
    if sorted(x["triangle_number"] for x in tris) != list(range(1, len(idx.triangles) + 1)):
        problems.append("експорт трикутників: номери не 1..M")
    if any(not 1 <= k <= n for x in tris for k in x["node_numbers"]):
        problems.append("експорт трикутників: node_numbers поза 1..N")
    return problems


def merge_inclusion_violations(case: MeshedCase) -> list[str]:
    """Одне merge-включення: сітка покриває область разом із включенням без накладань, трикутники
    не перетинають межу, вузол межі — 2 номери (зовні/всередині), обидва INTERFACE; підобласть
    трикутника (модель і експорт) = 1 всередині, 0 зовні; нумерація суцільна. Вузли розрізів
    перевіряє TestCutDoubleNumbering."""
    outer = case.imported.outer_boundary
    incl = [v for v, t in case.imported.inclusions if t == "merge"][0]
    ring = closed(incl)
    tris = case.tris
    problems = overlap_violations(analyze_conformity(tris, outer, case.holes, case.cuts))
    inside_area = sum(abs(cross(*t)) / 2.0 for t in tris if point_in_ring(centroid(t), incl))
    if abs(inside_area - abs(ring_area(incl))) > 1e-6 * abs(ring_area(incl)):
        problems.append(f"площа трикутників включення {inside_area:.3f} != площа включення {abs(ring_area(incl)):.3f}")
    crossing = [
        (p, q) for t in tris for p, q in tri_edges(t)
        if any(proper_cross(p, q, a, b) for a, b in zip(ring[:-1], ring[1:]))
    ]
    if crossing:
        problems.append(f"ребра трикутників перетинають межу включення: {len(crossing)}, напр. {crossing[:3]}")
    idx = case.mesh.indexed_mesh()
    on_ring = lambda xy: polyline_dist(xy, ring) < TOL  # noqa: E731
    problems += twin_violations(
        idx,
        on_interface=on_ring,
        side=lambda c: point_in_ring(c, incl),
        single_id=lambda xy: False,
        skip=lambda xy: not on_ring(xy) and case.cuts.polylines and case.cuts.contains(xy),
    )
    not_interface = [n for n in range(1, len(idx.nodes) + 1) if on_ring(_node_xy(idx, n)) and n not in idx.interface_nodes]
    if not_interface:
        problems.append(f"вузли межі включення не INTERFACE: {not_interface[:10]}")
    expected = [int(point_in_ring(centroid([_node_xy(idx, n) for n in t]), incl)) for t in idx.triangles]
    if idx.triangle_subdomains != expected:
        problems.append("triangle_subdomains не відповідає положенню трикутників відносно включення")
    with tempfile.TemporaryDirectory() as tmp:
        export_triangles(case.mesh, Path(tmp) / "triangles.json")
        export_nodes(idx, Path(tmp) / "nodes.json")
        exported = json.loads((Path(tmp) / "triangles.json").read_text(encoding="utf-8"))
        nodes = json.loads((Path(tmp) / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    for t in exported["triangles"]:
        c = centroid([tuple(xy) for xy in t["node_coordinates"]])
        if t["subdomain"] != int(point_in_ring(c, incl)):
            problems.append(f"експорт: трикутник {t['triangle_number']} має subdomain={t['subdomain']}")
            break
    if len(exported["interface_edges"]) != len(case.mesh.interface_edges) or not exported["interface_edges"]:
        problems.append("експорт: interface_edges порожній або неповний")
    for node in nodes:
        if on_ring(tuple(node["coordinates"])) and (node["attribute"] != "interface" or len(node["twins"]) != 1):
            problems.append(f"експорт: вузол межі {node['node_number']} {node['attribute']} twins={node['twins']}")
            break
    return problems + numbering_violations(case.mesh)


# ---------------------------------------------------------------- square_cut_offgrid у дочірньому процесі
def _child_main(name: str, mode: str) -> None:
    report = conformity(name, TriangulationMode(mode))
    print(json.dumps({k: (float(v) if isinstance(v, (int, float)) else [str(x) for x in v]) for k, v in report.items()}))


class _ChildRun:
    """Дочірній процес, який примусово вбивається таймером через OFFGRID_TIMEOUT_S."""

    def __init__(self, name: str, mode: TriangulationMode) -> None:
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"}
        self.timed_out = False
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", name, mode.value],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), env=env,
        )
        self._timer = threading.Timer(OFFGRID_TIMEOUT_S, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        self._result: dict | None = None

    def _on_timeout(self) -> None:
        if self.proc.poll() is None:
            self.timed_out = True
            self.proc.kill()

    def result(self) -> dict:
        if self._result is None:
            out, err = self.proc.communicate()  # таймер гарантує завершення не пізніше OFFGRID_TIMEOUT_S
            self._timer.cancel()
            lines = out.strip().splitlines()
            if self.timed_out:
                self._result = {"error": f"тріангуляція не завершилась за {OFFGRID_TIMEOUT_S:.0f} с"}
            elif self.proc.returncode == 0 and lines:
                self._result = json.loads(lines[-1])
            else:
                self._result = {"error": f"код {self.proc.returncode}: {err[-1500:]}"}
        return self._result

    def kill(self) -> None:
        self._timer.cancel()
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.communicate()


# ---------------------------------------------------------------- тести
HANDDRAWN = "fig22_handdrawn.json"
HANDDRAWN_FROM_BOUNDARY = "fig22_handdrawn_cut_from_boundary.json"


class TestNoOverlaps(unittest.TestCase):
    """П.1: сума площ = площа області; ребро не більше ніж у 2 трикутниках; внутрішності не перетинаються."""

    def _check(self, name: str, mode: TriangulationMode = BALANCED) -> None:
        self.assertEqual(overlap_violations(conformity(name, mode)), [])

    def test_fig22_like(self) -> None:
        self._check("fig22_like.json")

    def test_square_cut(self) -> None:
        self._check("square_cut.json")

    def test_square_cut_fast(self) -> None:
        self._check("square_cut.json", FAST)

    def test_square_inclusion(self) -> None:
        self._check("square_inclusion.json")

    def test_fig22_handdrawn(self) -> None:
        self._check(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary(self) -> None:
        self._check(HANDDRAWN_FROM_BOUNDARY)


class TestNoHangingNodes(unittest.TestCase):
    """П.2: вузол не лежить на стороні трикутника, вершиною якого не є."""

    def _check(self, name: str, mode: TriangulationMode = BALANCED) -> None:
        self.assertEqual(conformity(name, mode)["hanging"], [])

    def test_fig22_like(self) -> None:
        self._check("fig22_like.json")

    def test_square_cut(self) -> None:
        self._check("square_cut.json")

    def test_square_cut_fast(self) -> None:
        self._check("square_cut.json", FAST)

    def test_square_inclusion(self) -> None:
        self._check("square_inclusion.json")

    def test_fig22_handdrawn(self) -> None:
        self._check(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary(self) -> None:
        self._check(HANDDRAWN_FROM_BOUNDARY)


class TestNoInteriorValence3Nodes(unittest.TestCase):
    """П.3: немає внутрішніх вузлів валентності 3 ("трикутники в трикутниках")."""

    def _check(self, name: str) -> None:
        self.assertEqual(conformity(name)["valence3"], [])

    def test_fig22_like(self) -> None:
        self._check("fig22_like.json")

    def test_square_cut(self) -> None:
        self._check("square_cut.json")

    def test_square_inclusion(self) -> None:
        self._check("square_inclusion.json")

    def test_fig22_handdrawn(self) -> None:
        self._check(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary(self) -> None:
        self._check(HANDDRAWN_FROM_BOUNDARY)


class TestCutSidesConform(unittest.TestCase):
    """П.4: кожен вузол розрізу (крім вістря) — вершина трикутників з обох берегів; кожне ребро
    розрізу має по трикутнику з кожного берега; ребра сітки не перетинають розріз; розріз повністю
    покритий ребрами сітки."""

    def _check(self, name: str, mode: TriangulationMode = BALANCED) -> None:
        case = mesh_case(name, mode)
        self.assertTrue(case.cuts.polylines, "адаптер відкинув розрізи")
        self.assertEqual(cut_violations(conformity(name, mode)), [])

    def test_fig22_like(self) -> None:
        self._check("fig22_like.json")

    def test_square_cut(self) -> None:
        self._check("square_cut.json")

    def test_square_cut_fast(self) -> None:
        self._check("square_cut.json", FAST)

    def test_fig22_handdrawn(self) -> None:
        self._check(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary(self) -> None:
        self._check(HANDDRAWN_FROM_BOUNDARY)


class TestOffgridCut(unittest.TestCase):
    """П.4–5 для square_cut_offgrid (BALANCED і FAST): тріангуляція завершується за розумний час
    без накладань, береги розрізу конформні. Кожен режим — у дочірньому процесі з таймаутом."""

    runs: dict[TriangulationMode, _ChildRun]

    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = {mode: _ChildRun(OFFGRID_CASE, mode) for mode in (BALANCED, FAST)}

    @classmethod
    def tearDownClass(cls) -> None:
        for run in cls.runs.values():
            run.kill()

    def _result(self, mode: TriangulationMode) -> dict:
        result = self.runs[mode].result()
        self.assertNotIn("error", result, result.get("error"))
        return result

    def _check_terminates_without_overlaps(self, mode: TriangulationMode) -> None:
        result = self._result(mode)
        self.assertLessEqual(abs(result["sum_area"] - result["domain_area"]), 1e-6 * result["domain_area"])
        self.assertEqual(result["overlap_pairs"], [])
        self.assertEqual(result["edges_over_2"], [])
        self.assertEqual(result["hanging"], [])

    def _check_cut_sides(self, mode: TriangulationMode) -> None:
        result = self._result(mode)
        self.assertEqual({k: result[k] for k in CUT_KEYS if result[k]}, {})

    def test_terminates_without_overlaps_balanced(self) -> None:
        self._check_terminates_without_overlaps(BALANCED)

    def test_terminates_without_overlaps_fast(self) -> None:
        self._check_terminates_without_overlaps(FAST)

    def test_cut_sides_balanced(self) -> None:
        self._check_cut_sides(BALANCED)

    def test_cut_sides_fast(self) -> None:
        self._check_cut_sides(FAST)


class TestCutDoubleNumbering(unittest.TestCase):
    """П.6: внутрішній вузол розрізу — 2 номери з однаковими координатами, трикутники кожного
    берега посилаються на «свій»; вістря всередині області — 1 номер; кінець на зовнішній межі — 2.
    Решта вузлів (у т.ч. межа включення ignore) — 1 номер. Нумерація суцільна 1..N, 1..M."""

    def _check(self, name: str) -> None:
        case = mesh_case(name)
        self.assertTrue(case.cuts.polylines, "адаптер відкинув розрізи")
        problems = twin_violations(
            case.mesh.indexed_mesh(),
            on_interface=case.cuts.contains,
            side=case.cuts.side,
            single_id=lambda xy: pkey(xy) in case.cuts.interior_tips,
        )
        self.assertEqual(problems + numbering_violations(case.mesh), [])

    def test_fig22_like(self) -> None:
        self._check("fig22_like.json")

    def test_square_cut(self) -> None:
        self._check("square_cut.json")

    def test_fig22_handdrawn(self) -> None:
        self._check(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary(self) -> None:
        self._check(HANDDRAWN_FROM_BOUNDARY)


class TestInclusionNumbering(unittest.TestCase):
    """П.7: включення merge тріангулюється, кожен вузол його межі має 2 номери, трикутники включення
    не виходять за його межу, нумерація суцільна. Включення ignore — дірка з одинарними вузлами."""

    def test_square_inclusion_merge(self) -> None:
        self.assertEqual(merge_inclusion_violations(mesh_case("square_inclusion_merge.json")), [])

    def test_fig22_like_merge_with_cut(self) -> None:
        case = mesh_case("fig22_like_merge.json")
        self.assertTrue(case.cuts.polylines, "адаптер відкинув розрізи")
        self.assertEqual(merge_inclusion_violations(case), [])
        ring = closed([v for v, t in case.imported.inclusions if t == "merge"][0])
        problems = twin_violations(
            case.mesh.indexed_mesh(),
            on_interface=case.cuts.contains,
            side=case.cuts.side,
            single_id=lambda xy: pkey(xy) in case.cuts.interior_tips,
            skip=lambda xy: polyline_dist(xy, ring) < TOL,
        )
        self.assertEqual(problems + cut_violations(conformity("fig22_like_merge.json")), [])
        # Розріз лишається в cut_edges, межа включення — лише в interface_edges.
        cut_keys = {frozenset((pkey((a.x, a.y)), pkey((b.x, b.y)))) for a, b in case.mesh.cut_edges}
        iface_keys = {frozenset((pkey((a.x, a.y)), pkey((b.x, b.y)))) for a, b in case.mesh.interface_edges}
        self.assertTrue(cut_keys and iface_keys)
        self.assertFalse(cut_keys & iface_keys)

    def _check_ignore(self, name: str) -> None:
        case = mesh_case(name)
        rings = [closed(v) for v, t in case.imported.inclusions if t == "ignore"]
        self.assertTrue(rings)
        inside = [centroid(t) for t in case.tris if any(point_in_ring(centroid(t), r) for r in rings)]
        self.assertEqual(inside, [], "трикутники всередині включення ignore")
        groups = _coordinate_groups(case.mesh.indexed_mesh())
        doubled = [xy for xy, ids in groups.items() if len(ids) > 1 and any(polyline_dist(xy, r) < TOL for r in rings)]
        self.assertEqual(doubled, [])
        self.assertEqual(numbering_violations(case.mesh), [])
        # Межа дірки — межа області: BOUNDARY; вузли розрізу на ній лишаються INTERFACE.
        idx = case.mesh.indexed_mesh()
        on_hole = lambda xy: any(polyline_dist(xy, r) < TOL for r in rings)  # noqa: E731
        on_cut = lambda xy: bool(case.cuts.polylines) and case.cuts.contains(xy)  # noqa: E731
        wrong = [
            n for n in range(1, len(idx.nodes) + 1)
            if on_hole(_node_xy(idx, n))
            and (n in idx.interface_nodes) != on_cut(_node_xy(idx, n))
        ]
        self.assertEqual(wrong, [], "вузли межі дірки: BOUNDARY, крім вузлів розрізу (INTERFACE)")
        self.assertTrue(all(n in idx.boundary_nodes for n in range(1, len(idx.nodes) + 1) if on_hole(_node_xy(idx, n))))
        with tempfile.TemporaryDirectory() as tmp:
            export_nodes(idx, Path(tmp) / "nodes.json")
            nodes = json.loads((Path(tmp) / "nodes.json").read_text(encoding="utf-8"))["nodes"]
        for node in nodes:
            xy = tuple(node["coordinates"])
            if on_hole(xy) and not on_cut(xy):
                self.assertEqual((node["attribute"], node["on_boundary"], node["twins"]), ("boundary", True, []))

    def test_square_inclusion_ignore(self) -> None:
        self._check_ignore("square_inclusion.json")

    def test_fig22_like_ignore(self) -> None:
        self._check_ignore("fig22_like.json")

    def test_fig22_handdrawn_ignore(self) -> None:
        self._check_ignore(HANDDRAWN)

    def test_fig22_handdrawn_cut_from_boundary_ignore(self) -> None:
        self._check_ignore(HANDDRAWN_FROM_BOUNDARY)


class TestInclusionTypeSelection(unittest.TestCase):
    """Тип включення доходить до мешера: з JSON — для кожного включення окремо; для намальованої
    геометрії (без source_contours) — один inclusion_mode для всіх."""

    SHELL = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    LEFT = [(60.0, 80.0), (160.0, 80.0), (160.0, 220.0), (60.0, 220.0)]
    RIGHT = [(240.0, 80.0), (340.0, 80.0), (340.0, 220.0), (240.0, 220.0)]

    def _run(self, source: MeshSourceContours | None, inclusion_mode: str = "ignore") -> IndexedMesh:
        contours = [[*r, r[0]] for r in (self.SHELL, self.LEFT, self.RIGHT)]
        mesh, _ = adapter_module.TriangulationAdapter().run_from_contours(
            contours, mode=FAST, source_contours=source, inclusion_mode=inclusion_mode
        )
        return mesh.indexed_mesh()

    def _subdomain_area(self, idx: IndexedMesh, ring) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for pos, t in enumerate(idx.triangles):
            pts = [_node_xy(idx, n) for n in t]
            if point_in_ring(centroid(pts), ring):
                out[idx.triangle_subdomain(pos)] += abs(cross(*pts)) / 2.0
        return dict(out)

    def test_json_types_per_inclusion(self) -> None:
        source = MeshSourceContours(
            outer_boundary=self.SHELL, inclusions=[(self.LEFT, "ignore"), (self.RIGHT, "merge")], cuts=[]
        )
        idx = self._run(source, inclusion_mode="ignore")
        self.assertEqual(self._subdomain_area(idx, self.LEFT), {})  # дірка
        right = self._subdomain_area(idx, self.RIGHT)
        self.assertEqual(list(right), [1])
        self.assertAlmostEqual(right[1], 100.0 * 140.0, delta=1e-6)
        groups = _coordinate_groups(idx)
        on = lambda ring: [ids for xy, ids in groups.items() if polyline_dist(xy, closed(ring)) < TOL]  # noqa: E731
        self.assertTrue(all(len(ids) == 1 for ids in on(self.LEFT)))
        self.assertTrue(on(self.RIGHT) and all(len(ids) == 2 for ids in on(self.RIGHT)))

    def test_drawn_geometry_uses_one_mode(self) -> None:
        ignored = self._run(None, inclusion_mode="ignore")
        self.assertEqual(set(ignored.triangle_subdomains or [0]), {0})
        self.assertEqual(self._subdomain_area(ignored, self.LEFT), {})
        merged = self._run(None, inclusion_mode="merge")
        self.assertEqual(set(merged.triangle_subdomains), {0, 1, 2})
        areas = {**self._subdomain_area(merged, self.LEFT), **self._subdomain_area(merged, self.RIGHT)}
        self.assertEqual(sorted(areas), [1, 2])
        for area in areas.values():
            self.assertAlmostEqual(area, 100.0 * 140.0, delta=1e-6)

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._run(None, inclusion_mode="glue")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _child_main(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
