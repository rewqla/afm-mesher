"""Еталон сітки й заміри часу по етапах для набору кейсів × режимів.

Запуск з кореня проєкту:
    .venv\\Scripts\\python diagnostics\\perf_reference.py --save     # зберегти еталон (до оптимізацій)
    .venv\\Scripts\\python diagnostics\\perf_reference.py            # порівняти з еталоном + таблиця часу
    .venv\\Scripts\\python diagnostics\\perf_reference.py --repeat 3 --only fig22_like --mode accurate

Хеш покриває: вузли (координати, округлені до 1e-9) у порядку нумерації, трикутники (номери вузлів),
boundary/interface-вузли, двійники, підобласті, bandwidth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.application.services.advancing_front_mesher as afm_module  # noqa: E402
import src.presentation.tri_debug as tri_debug_module  # noqa: E402
from src.infrastructure.io.coordinate_json_importer import CoordinateJsonImporter  # noqa: E402
from src.presentation.tools import TriangulationMode  # noqa: E402
from src.presentation.triangulation_adapter import TriangulationAdapter  # noqa: E402

OUTPUT_DIR = ROOT / "diagnostics" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REFERENCE_PATH = OUTPUT_DIR / "perf_reference.json"
afm_module._AFM_DEBUG_LOG_PATH = OUTPUT_DIR / "perf_debug.log"
tri_debug_module._LOG_PATH = OUTPUT_DIR / "perf_debug.log"

CASES = {
    "fig22_like": ROOT / "diagnostics/cases/fig22_like.json",
    "fig22_like_merge": ROOT / "diagnostics/cases/fig22_like_merge.json",
    "square_cut_offgrid": ROOT / "diagnostics/cases/square_cut_offgrid.json",
    "square_inclusion": ROOT / "diagnostics/cases/square_inclusion.json",
    "complex_region_with_merge": ROOT / "tests/fixtures/complex_region_with_merge.json",
    "complex_nested_inclusions": ROOT / "tests/fixtures/complex_nested_inclusions.json",
    "fig22_handdrawn": ROOT / "diagnostics/cases/fig22_handdrawn.json",
    "img_spot_with_circle": ROOT / "data/images/spot_with_circle.png",
}
MODES = [TriangulationMode.FAST, TriangulationMode.BALANCED, TriangulationMode.ACCURATE]


def run(name: str, mode: TriangulationMode):
    path = CASES[name]
    adapter = TriangulationAdapter()
    if path.suffix == ".png":
        from PySide6.QtGui import QImage

        image = QImage(str(path))
        return adapter.run(image, mode=mode)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "contours" in raw:
        return adapter.run_from_contours(raw["contours"], mode=mode)
    importer = CoordinateJsonImporter()
    imported = importer.load(str(path))
    # Як у diagnose_mesh: координати як намальовані контури.
    return adapter.run_from_contours(importer.to_canvas_contours(imported), mode=mode)


def mesh_signature(mesh) -> dict:
    idx = mesh.indexed_mesh()
    payload = {
        "nodes": [(round(x, 9), round(y, 9)) for x, y in idx.nodes],
        "triangles": [tuple(t) for t in idx.triangles],
        "boundary": sorted(idx.boundary_nodes),
        "interface": sorted(idx.interface_nodes),
        "twins": [tuple(g) for g in idx.twin_groups],
        "subdomains": list(idx.triangle_subdomains),
        "bandwidth": idx.bandwidth,
    }
    digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    return {
        "nodes": len(idx.nodes),
        "triangles": len(idx.triangles),
        "bandwidth": idx.bandwidth,
        "hash": digest,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="кращий з N прогонів для часу")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--mode", nargs="*", default=None)
    parser.add_argument("--json-out", default=None, help="зберегти таймінги у файл")
    args = parser.parse_args()

    if any(CASES[n].suffix == ".png" for n in (args.only or CASES)):
        from PySide6.QtGui import QGuiApplication

        _app = QGuiApplication.instance() or QGuiApplication(["perf"])  # noqa: F841

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8")) if REFERENCE_PATH.exists() else {}
    modes = [TriangulationMode(m) for m in args.mode] if args.mode else MODES
    results: dict[str, dict] = {}
    matched = mismatched = 0
    print(f"{'case':28} {'mode':9} {'nodes':>6} {'tris':>6} {'bw':>5} "
          f"{'total':>8} {'contour':>8} {'afm':>8} {'post':>8} {'numb':>8}  ref")
    for name in args.only or CASES:
        for mode in modes:
            best = None
            sig = None
            for _ in range(max(1, args.repeat)):
                t0 = perf_counter()
                mesh, _q = run(name, mode)
                wall = perf_counter() - t0
                t = mesh.timings
                row = {"wall": wall, "total": t.total_s, "contours": t.contours_s, "afm": t.afm_s,
                       "post": t.postprocess_s, "numbering": t.numbering_s}
                if best is None or row["total"] < best["total"]:
                    best = row
                sig = mesh_signature(mesh)
            key = f"{name}/{mode.value}"
            results[key] = {"signature": sig, "timings": best}
            status = ""
            if not args.save and key in reference:
                ok = reference[key]["signature"] == sig
                matched += ok
                mismatched += not ok
                status = "OK" if ok else "MISMATCH"
            print(f"{name:28} {mode.value:9} {sig['nodes']:6} {sig['triangles']:6} {sig['bandwidth']:5} "
                  f"{best['total']:8.3f} {best['contours']:8.3f} {best['afm']:8.3f} {best['post']:8.3f} "
                  f"{best['numbering']:8.3f}  {status}", flush=True)

    if args.save:
        merged = {**reference, **results}
        REFERENCE_PATH.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        print(f"еталон збережено: {REFERENCE_PATH} ({len(results)} комбінацій)")
    else:
        print(f"збіг з еталоном: {matched}, розбіжностей: {mismatched}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    if mismatched:
        sys.exit(1)


if __name__ == "__main__":
    main()
