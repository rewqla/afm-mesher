"""Діагностика сітки: накладання, висячі вузли, неузгодженість берегів розрізу, вироджені трикутники.

Запуск з кореня проєкту:
    .venv\\Scripts\\python diagnostics\\diagnose_mesh.py                 # усі кейси з diagnostics/cases
    .venv\\Scripts\\python diagnostics\\diagnose_mesh.py fig22_like.json  # один кейс
    .venv\\Scripts\\python diagnostics\\diagnose_mesh.py --mode fast --top 20

Код у src/ не змінюється: мешер лише обгортається підкласом, щоб позначити
походження кожного трикутника (звичайний AFM-крок / reversed-edge recovery / ear-clip fallback)
і зберегти сиру сітку до згладжування. Чистий Python, без shapely.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.tri_debug as tri_debug_module  # noqa: E402
import src.presentation.triangulation_adapter as adapter_module  # noqa: E402
from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
afm_module._AFM_DEBUG_LOG_PATH = OUTPUT_DIR / "afm_debug.log"
tri_debug_module._LOG_PATH = OUTPUT_DIR / "afm_debug.log"

KEY = 6
TOL = 1e-6


def pkey(p) -> tuple[float, float]:
    return (round(p[0], KEY), round(p[1], KEY))


# ---------------------------------------------------------------- instrumented mesher
class InstrumentedMesher(afm_module.AdvancingFrontMesher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.origin: dict[frozenset, str] = {}
        self.input_boundary = []
        self.input_holes = []
        self.input_cuts = []
        self.pass_results: list[dict] = []
        self.smoothed_to_raw: list[tuple] = []
        self._pending_raw = None
        self.events: Counter = Counter()

    def generate_with_holes_and_cuts(self, boundary, holes, cuts):
        self.input_boundary = list(boundary)
        self.input_holes = [list(h) for h in holes]
        self.input_cuts = [list(c) for c in cuts]
        return super().generate_with_holes_and_cuts(boundary, holes, cuts)

    def _generate_single_pass(self, polygon, holes, cut_segments, target_h):
        try:
            result = super()._generate_single_pass(polygon, holes, cut_segments, target_h)
        except ValueError as error:
            self.pass_results.append({"target_h": target_h, "error": str(error)})
            raise
        self.pass_results.append({"target_h": target_h, "error": None})
        self._pending_raw = result
        return result

    def smooth(self, mesh, boundary, interface_points=None, iterations=5):
        out = super().smooth(mesh, boundary, interface_points, iterations)
        self.smoothed_to_raw.append((out, self._pending_raw))
        return out

    def _find_advancement(self, front, polygon, holes, cut_segments, target_step):
        result = super()._find_advancement(front, polygon, holes, cut_segments, target_step)
        if result is not None:
            idx, a, b, c = result
            tag = "afm_reversed_edge" if front[idx] != (a, b) else "afm"
            self.events[tag] += 1
            self.origin[frozenset(pkey((p.x, p.y)) for p in (a, b, c))] = tag
        return result

    def _fallback_triangulate_front(self, front, polygon, holes, cut_segments):
        loop = self._order_front_loop(front)
        tris = super()._fallback_triangulate_front(front, polygon, holes, cut_segments)
        self.events["fallback_calls"] += 1
        self.events["fallback_front_edges"] += len(front)
        self.events["fallback_loop_points"] += len(loop) if loop else 0
        self.events["fallback_triangles"] += len(tris)
        for t in tris:
            self.origin[frozenset(pkey((p.x, p.y)) for p in (t.a, t.b, t.c))] = "fallback_ear_clip"
        return tris


# ---------------------------------------------------------------- geometry (pure python)
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
    """Строгий перетин; кінці ближче dist_eps до іншого відрізка вважаються дотиком (ключі округлені до 1e-6)."""
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


def quality(a, b, c) -> float:
    ar = abs(cross(a, b, c)) / 2.0
    s = sum((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for p, q in ((a, b), (b, c), (c, a)))
    return 4.0 * math.sqrt(3.0) * ar / s if s > 0 else 0.0


def min_angle_deg(a, b, c) -> float:
    def ang(p, q, r):
        v1 = (q[0] - p[0], q[1] - p[1])
        v2 = (r[0] - p[0], r[1] - p[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))))

    return min(ang(a, b, c), ang(b, c, a), ang(c, a, b))


def centroid(t):
    return ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0)


def fmt(p) -> str:
    return f"({p[0]:.1f}, {p[1]:.1f})"


# ---------------------------------------------------------------- checks
@dataclass
class Report:
    name: str
    lines: list[str] = field(default_factory=list)
    counts: dict[str, object] = field(default_factory=dict)

    def add(self, text: str) -> None:
        self.lines.append(text)


def check_mesh(tris, shell, holes, cuts, origin, top, report: Report, prefix: str) -> None:
    n = len(tris)
    areas = [cross(*t) / 2.0 for t in tris]

    def otag(i: int) -> str:
        if origin is None:
            return ""
        return f" [{origin.get(frozenset(pkey(p) for p in tris[i]), '?')}]"

    def in_domain(p) -> bool:
        return point_in_ring(p, shell) and not any(point_in_ring(p, h) for h in holes)

    scale = max(math.dist(a, b) for t in tris for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    grid = Grid(scale)
    for i, t in enumerate(tris):
        grid.insert(i, t)

    # 1) перетин внутрішностей
    overlap_pairs, nested = [], []
    for i, t in enumerate(tris):
        for j in grid.query(t):
            if j <= i:
                continue
            inter = tri_intersection_area(t, tris[j])
            smaller = min(abs(areas[i]), abs(areas[j]))
            if inter > 1e-9 and inter > 1e-6 * smaller:
                overlap_pairs.append((i, j, inter))
                if smaller > 0 and inter >= 0.98 * smaller:
                    nested.append((i, j, inter))
    overlap_area = sum(x[2] for x in overlap_pairs)

    # 2) площі
    domain_area = abs(ring_area(shell)) - sum(abs(ring_area(h)) for h in holes)
    sum_area = sum(abs(a) for a in areas)
    outside = [i for i, t in enumerate(tris) if not in_domain(centroid(t))]
    report.add(f"  {prefix} трикутників={n}, вузлів(унік. коорд.)={len({pkey(p) for t in tris for p in t})}")
    report.add(
        f"  {prefix} площа області={domain_area:.1f}, сума площ трикутників={sum_area:.1f}, "
        f"площа накладань={overlap_area:.2f}, непокрито≈{domain_area - (sum_area - overlap_area):.2f}"
    )
    inverted = [i for i, a in enumerate(areas) if a <= 1e-12]

    # 3) ребра
    edge_tris: dict[frozenset, list[int]] = defaultdict(list)
    for i, t in enumerate(tris):
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_tris[frozenset((pkey(p), pkey(q)))].append(i)
    edges = list(edge_tris)
    over2 = [(e, ts) for e, ts in edge_tris.items() if len(ts) > 2]

    rings = [[*r, r[0]] for r in (shell, *holes)]

    def on_rings(p) -> bool:
        return any(polyline_dist(p, r) < TOL for r in rings)

    def edge_mid(e):
        p, q = tuple(e)
        return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)

    def edge_on_polyline(e, line) -> bool:
        return all(polyline_dist(x, line) < TOL for x in (*e, edge_mid(e)))

    open_edges = [
        e for e, ts in edge_tris.items()
        if len(ts) == 1 and not all(on_rings(x) for x in (*e, edge_mid(e)))
        and not any(edge_on_polyline(e, c) for c in cuts)
    ]

    # 4) висячі вузли
    egrid = Grid(scale)
    for k, e in enumerate(edges):
        egrid.insert(k, list(e))
    vertices = sorted({pkey(p) for t in tris for p in t})
    hanging = []
    for v in vertices:
        for k in egrid.query([v]):
            e = edges[k]
            if v in e:
                continue
            p, q = tuple(e)
            L = math.dist(p, q)
            if L == 0:
                continue
            t = ((v[0] - p[0]) * (q[0] - p[0]) + (v[1] - p[1]) * (q[1] - p[1])) / (L * L)
            if 1e-9 < t < 1 - 1e-9 and seg_dist(v, p, q) <= TOL * max(L, 1.0):
                hanging.append((v, e))

    # 5) якість
    qs = [quality(*t) for t in tris]
    med_area = sorted(abs(a) for a in areas)[n // 2] if n else 0.0
    low_q = [i for i, q in enumerate(qs) if q < 0.1]
    small_angle = [i for i, t in enumerate(tris) if min_angle_deg(*t) < 10.0]
    tiny = [i for i, a in enumerate(areas) if abs(a) < 0.05 * med_area]

    incident = defaultdict(list)
    for i, t in enumerate(tris):
        for p in t:
            incident[pkey(p)].append(i)

    # 5b) внутрішні вузли валентності 3: візуально "трикутник, поділений на 3 дрібні всередині"
    def on_constraint(v) -> bool:
        return on_rings(v) or any(polyline_dist(v, c) < TOL for c in cuts)

    valence3 = [v for v in vertices if len(incident[v]) == 3 and not on_constraint(v)]

    # 6) розрізи
    cut_issues: dict[str, list] = defaultdict(list)
    cut_scalar: dict[str, float] = {}
    for ci, cut in enumerate(cuts):
        tips = [cut[0], cut[-1]]
        segs = list(zip(cut[:-1], cut[1:]))

        def side(pt, segs=segs) -> int:
            a, b = min(segs, key=lambda s: seg_dist(pt, *s))
            return 1 if cross(a, b, pt) > 0 else -1

        for tp in tips:
            onb = polyline_dist(tp, rings[0]) < TOL
            report.add(f"  {prefix} розріз #{ci}: кінець {fmt(tp)} {'на зовнішній межі' if onb else 'всередині області'}")
        for v in vertices:
            if polyline_dist(v, cut) >= TOL or any(math.dist(v, tp) < TOL for tp in tips):
                continue
            if len({side(centroid(tris[i])) for i in incident[v]}) < 2:
                cut_issues["вузол розрізу є вершиною трикутників лише з одного берега"].append(v)
        cut_edge_len = 0.0
        for e in edges:
            if not edge_on_polyline(e, cut):
                continue
            p, q = tuple(e)
            cut_edge_len += math.dist(p, q)
            if sorted(side(centroid(tris[i])) for i in edge_tris[e]) != [-1, 1]:
                cut_issues["ребро розрізу без пари трикутників (по одному з кожного берега)"].append(edge_mid(e))
        cut_len = sum(math.dist(a, b) for a, b in segs)
        if abs(cut_len - cut_edge_len) > 1e-3:
            cut_scalar[f"розріз #{ci}: довжина {cut_len:.1f}, покрита ребрами сітки {cut_edge_len:.1f}"] = round(
                cut_len - cut_edge_len, 3)
        for e in edges:
            p, q = tuple(e)
            if any(proper_cross(p, q, a, b) for a, b in segs):
                cut_issues["ребро сітки перетинає розріз"].append(edge_mid(e))

    c = report.counts
    c[f"{prefix}overlap_area"] = round(overlap_area, 3)
    c[f"{prefix}overlap_pairs"] = len(overlap_pairs)
    c[f"{prefix}nested"] = len(nested)
    c[f"{prefix}edges>2"] = len(over2)
    c[f"{prefix}open_edges"] = len(open_edges)
    c[f"{prefix}outside"] = len(outside)
    c[f"{prefix}inverted"] = len(inverted)
    c[f"{prefix}hanging"] = len(hanging)
    c[f"{prefix}q<0.1"] = len(low_q)
    c[f"{prefix}angle<10"] = len(small_angle)
    c[f"{prefix}tiny"] = len(tiny)
    c[f"{prefix}valence3"] = len(valence3)
    for k, v in cut_issues.items():
        c[f"{prefix}{k}"] = len(v)
    for k, v in cut_scalar.items():
        c[f"{prefix}{k}"] = v

    def section(title, items, render) -> None:
        report.add(f"  {prefix} {title}: {len(items)}")
        for it in items[:top]:
            report.add(f"      - {render(it)}")
        if len(items) > top:
            report.add(f"      ... ще {len(items) - top}")

    section("перетини внутрішностей трикутників (пари)", overlap_pairs,
            lambda x: f"T{x[0]}{fmt(centroid(tris[x[0]]))}{otag(x[0])} ∩ T{x[1]}{fmt(centroid(tris[x[1]]))}{otag(x[1])} "
                      f"площа={x[2]:.2f}")
    section("вкладені трикутники (менший ≥98% всередині іншого)", nested,
            lambda x: f"T{x[0]}{otag(x[0])} / T{x[1]}{otag(x[1])} біля {fmt(centroid(tris[x[1]]))}")
    section("ребра, що належать >2 трикутникам", over2, lambda x: f"{' - '.join(fmt(p) for p in x[0])} ×{len(x[1])}")
    section("відкриті ребра всередині області (дірки в покритті)", open_edges, lambda e: " - ".join(fmt(p) for p in e))
    section("трикутники з центроїдом поза областю", outside, lambda i: f"T{i} {fmt(centroid(tris[i]))}{otag(i)}")
    section("інвертовані/вироджені (орієнт. площа<=0)", inverted, lambda i: f"T{i} {fmt(centroid(tris[i]))}{otag(i)}")
    section("висячі вузли (вузол лежить на стороні трикутника, вершиною якого не є)", hanging,
            lambda x: f"вузол {fmt(x[0])} на ребрі {' - '.join(fmt(p) for p in x[1])}")
    section("якість q<0.1", low_q, lambda i: f"T{i} {fmt(centroid(tris[i]))} q={qs[i]:.3f}{otag(i)}")
    section("мін. кут <10°", small_angle, lambda i: f"T{i} {fmt(centroid(tris[i]))}{otag(i)}")
    section("аномально дрібні (площа < 5% медіани)", tiny,
            lambda i: f"T{i} {fmt(centroid(tris[i]))} площа={abs(areas[i]):.2f} (медіана {med_area:.1f}){otag(i)}")
    section("внутрішні вузли валентності 3 (3 трикутники всередині більшого)", valence3,
            lambda v: f"вузол {fmt(v)}")
    for k, v in cut_issues.items():
        section(k, v, fmt)
    for k, v in cut_scalar.items():
        report.add(f"  {prefix} {k} -> НЕ покрито {v}")


# ---------------------------------------------------------------- runner
def run_case(path: Path, mode: TriangulationMode, top: int) -> Report:
    importer = CoordinateJsonImporter()
    imported = importer.load(str(path))
    contours = importer.to_canvas_contours(imported)

    holder: dict[str, InstrumentedMesher] = {}

    class _Factory(InstrumentedMesher):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["m"] = self

    adapter_module.AdvancingFrontMesher = _Factory
    try:
        mesh, coefficient = adapter_module.TriangulationAdapter().run_from_contours(contours, mode=mode)
    finally:
        adapter_module.AdvancingFrontMesher = afm_module.AdvancingFrontMesher

    m = holder["m"]
    report = Report(path.name)
    report.add(f"=== {path.name}  (режим {mode.value}) ===")
    report.add(f"  середня якість={coefficient:.3f}")
    report.add("  проходи AFM: " + "; ".join(
        f"h={r['target_h']:.1f}:{'OK' if r['error'] is None else 'FAIL(' + r['error'] + ')'}" for r in m.pass_results))
    report.add(f"  події мешера (сумарно по всіх проходах): {dict(m.events)}")
    report.add(
        f"  на вході мешера: межа {len(m.input_boundary)} точок, дірок {len(m.input_holes)}, розрізів {len(m.input_cuts)}"
        + (" (розрізи ВІДКИНУТО адаптером!)" if imported.cuts and not m.input_cuts else "")
    )

    shell = [(p.x, p.y) for p in m.input_boundary]
    holes = [[(p.x, p.y) for p in h] for h in m.input_holes]
    cuts = [[(p.x, p.y) for p in c] for c in m.input_cuts]

    raw = next((r for out, r in m.smoothed_to_raw if out.triangles is mesh.triangles), None)
    if raw is not None:
        raw_tris = [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in raw[0].triangles]
        origin_counts = Counter(m.origin.get(frozenset(pkey(p) for p in t), "?") for t in raw_tris)
        report.add(f"  походження трикутників обраного проходу: {dict(origin_counts)}")
        report.add("  --- СИРА сітка обраного проходу (до згладжування) ---")
        check_mesh(raw_tris, shell, holes, cuts, m.origin, top, report, "[raw]")
    else:
        report.add("  (сира сітка недоступна: спрацював ear-clip fallback для простої області)")

    final_tris = [((t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)) for t in mesh.triangles]
    report.add("  --- ФІНАЛЬНА сітка (після згладжування, те, що бачить UI) ---")
    check_mesh(final_tris, shell, holes, cuts, None, top, report, "[final]")

    idx = mesh.indexed_mesh()
    uniq = len({pkey(p) for p in idx.nodes})
    report.add(f"  нумерація IndexedMesh: вузлів={len(idx.nodes)}, унікальних координат={uniq}, "
               f"двійників={len(idx.nodes) - uniq}, boundary={len(idx.boundary_nodes)}, interface={len(idx.interface_nodes)}")
    return report


def set_log_path(path: Path) -> None:
    afm_module._AFM_DEBUG_LOG_PATH = path
    tri_debug_module._LOG_PATH = path


def log_path_for(case: Path) -> Path:
    return OUTPUT_DIR / f"afm_debug_{case.stem}.log"


def run_child(case: Path, mode: TriangulationMode, top: int) -> None:
    log = log_path_for(case)
    log.write_text("", encoding="utf-8")
    set_log_path(log)
    try:
        rep = run_case(case, mode, top)
    except Exception as exc:  # noqa: BLE001
        print(f"=== {case.name}: ПОМИЛКА {type(exc).__name__}: {exc}\n")
        return
    print("\n".join(rep.lines))
    print("  ПІДСУМОК (лише ненульові):")
    for k, v in rep.counts.items():
        if v:
            print(f"    {k}: {v}")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="імена файлів у diagnostics/cases або шляхи до JSON")
    parser.add_argument("--mode", default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--top", type=int, default=10, help="скільки координат показувати на кожен тип проблеми")
    parser.add_argument("--timeout", type=float, default=60.0, help="таймаут на кейс, с")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    paths = [Path(c) if Path(c).exists() else CASES_DIR / c for c in args.cases] or sorted(CASES_DIR.glob("*.json"))
    mode = TriangulationMode(args.mode)
    if args.child:
        run_child(paths[0], mode, args.top)
        return

    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"}
    for case in paths:
        cmd = [sys.executable, str(Path(__file__).resolve()), str(case), "--mode", args.mode,
               "--top", str(args.top), "--child"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=args.timeout, env=env, cwd=str(ROOT))
            print(proc.stdout, end="")
            if proc.returncode != 0:
                print(f"=== {case.name}: процес завершився з кодом {proc.returncode}\n{proc.stderr[-2000:]}")
        except subprocess.TimeoutExpired:
            print(f"=== {case.name}  (режим {args.mode}) ===")
            print(f"  TIMEOUT {args.timeout:.0f} с. Останні рядки {log_path_for(case).name}:")
            try:
                tail = log_path_for(case).read_text(encoding="utf-8").splitlines()[-15:]
            except OSError:
                tail = []
            for line in tail:
                print(f"    {line[:300]}")
            print()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
