from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.services import advancing_front_mesher as afm_module
from src.application.services.mesh_postprocessing import renumber_nodes_rcm, renumber_nodes_rcm_multistart
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary


def main() -> None:
    args = _parse_args()
    image_paths = args.image or [str(Path("data") / "images" / "one.png")]
    grid_sizes = args.grid_size or [1000, 1500, 2000]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    for image_path in image_paths:
        cases.append(_build_image_case(Path(image_path)))
    for node_count in grid_sizes:
        cases.append(_build_grid_case(node_count))

    rows: list[dict[str, object]] = []
    for case in cases:
        nodes = case["nodes"]  # type: ignore[assignment]
        triangles = case["triangles"]  # type: ignore[assignment]
        case_name = str(case["name"])
        max_candidates = _candidate_cap(len(nodes), args.candidate_mode, args.fixed_max_candidates)

        baseline_times: list[float] = []
        multistart_times: list[float] = []
        baseline_beta = None
        multistart_beta = None
        baseline_bandwidth = None
        multistart_bandwidth = None

        for _ in range(args.runs):
            t0 = time.perf_counter()
            _, _, baseline_bandwidth = renumber_nodes_rcm(nodes, triangles)
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            _, _, multistart_bandwidth = renumber_nodes_rcm_multistart(
                nodes,
                triangles,
                max_candidates=max_candidates,
            )
            t1 = time.perf_counter()
            multistart_times.append((t1 - t0) * 1000.0)

            if baseline_beta is None:
                baseline_beta = baseline_bandwidth
            if multistart_beta is None:
                multistart_beta = multistart_bandwidth

        row = {
            "case": case_name,
            "kind": case["kind"],
            "nodes": len(nodes),
            "triangles": len(triangles),
            "candidate_mode": args.candidate_mode,
            "max_candidates": max_candidates,
            "baseline_beta": baseline_beta,
            "multistart_beta": multistart_beta,
            "delta_beta": (baseline_beta - multistart_beta) if baseline_beta is not None and multistart_beta is not None else None,
            "baseline_time_ms": statistics.median(baseline_times),
            "multistart_time_ms": statistics.median(multistart_times),
            "time_overhead_ms": statistics.median(multistart_times) - statistics.median(baseline_times),
        }
        rows.append(row)
        print(
            f"{case_name}: nodes={row['nodes']} triangles={row['triangles']} "
            f"beta={row['multistart_beta']} baseline_ms={row['baseline_time_ms']:.3f} "
            f"multistart_ms={row['multistart_time_ms']:.3f} cap={max_candidates}"
        )

    _write_reports(output_dir, rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile RCM multistart scaling on image and synthetic graph cases.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path to reconstruct a raw AFM graph before RCM.",
    )
    parser.add_argument(
        "--grid-size",
        action="append",
        type=int,
        default=[],
        help="Synthetic rectangular grid size in nodes.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("fixed", "sqrt", "log"),
        default="fixed",
        help="How to derive the multistart candidate cap from N.",
    )
    parser.add_argument(
        "--fixed-max-candidates",
        type=int,
        default=20,
        help="Candidate cap when --candidate-mode=fixed.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measurements to aggregate per case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "output",
        help="Directory for CSV/JSON reports.",
    )
    return parser.parse_args()


def _build_image_case(image_path: Path) -> dict[str, object]:
    resolved = _resolve_image_path(image_path)
    preprocessed = preprocess_photo_to_boundary(resolved, threshold=127, epsilon=2.0)
    boundary = preprocessed.boundary
    mesher = afm_module.AdvancingFrontMesher(
        min_triangle_quality=0.01,
        max_iterations_factor=180,
        target_edge_length=24.0,
        smoothing_iterations=2,
        numbering_strategy="rcm_multistart",
    )
    captured: dict[str, object] = {}

    original_rcm = afm_module.renumber_nodes_rcm_multistart

    def _capturing_rcm(
        nodes: list[tuple[float, float]],
        triangles: list[tuple[int, int, int]],
        candidates: list[int] | None = None,
        max_candidates: int = 20,
        probe_stride: int = 12,
    ) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
        captured["nodes"] = list(nodes)
        captured["triangles"] = [tuple(triangle) for triangle in triangles]
        return list(nodes), [tuple(triangle) for triangle in triangles], 0

    afm_module.renumber_nodes_rcm_multistart = _capturing_rcm  # type: ignore[assignment]
    try:
        mesher.generate(boundary)
    finally:
        afm_module.renumber_nodes_rcm_multistart = original_rcm  # type: ignore[assignment]

    nodes = captured.get("nodes")
    triangles = captured.get("triangles")
    if not isinstance(nodes, list) or not isinstance(triangles, list):
        raise SystemExit(f"Failed to capture raw graph from image: {resolved}")

    return {
        "kind": "image",
        "name": resolved.name,
        "nodes": [tuple(node) for node in nodes],
        "triangles": [tuple(triangle) for triangle in triangles],
    }


def _build_grid_case(node_count: int) -> dict[str, object]:
    rows, cols = _factor_grid(node_count)
    nodes, triangles = _build_rectangular_triangulation(rows, cols)
    return {
        "kind": "grid",
        "name": f"grid_{rows}x{cols}",
        "nodes": nodes,
        "triangles": triangles,
    }


def _factor_grid(node_count: int) -> tuple[int, int]:
    if node_count < 4:
        raise SystemExit("grid size must be at least 4 nodes")
    root = int(math.sqrt(node_count))
    best_rows, best_cols = 1, node_count
    best_score = float("inf")
    for rows in range(1, root + 1):
        cols = math.ceil(node_count / rows)
        score = abs(rows - cols) + (rows * cols - node_count) / max(node_count, 1)
        if score < best_score:
            best_score = score
            best_rows, best_cols = rows, cols
    return best_rows, best_cols


def _build_rectangular_triangulation(rows: int, cols: int) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    nodes = [(float(col), float(row)) for row in range(rows) for col in range(cols)]
    triangles: list[tuple[int, int, int]] = []

    for row in range(rows - 1):
        row_offset = row * cols
        next_row_offset = (row + 1) * cols
        for col in range(cols - 1):
            top_left = row_offset + col
            top_right = top_left + 1
            bottom_left = next_row_offset + col
            bottom_right = bottom_left + 1
            triangles.append((top_left, top_right, bottom_right))
            triangles.append((top_left, bottom_right, bottom_left))

    return nodes, triangles


def _candidate_cap(node_count: int, mode: str, fixed_max_candidates: int) -> int:
    if mode == "fixed":
        return max(1, fixed_max_candidates)
    if mode == "sqrt":
        return max(1, math.ceil(math.sqrt(node_count)))
    if mode == "log":
        return max(1, math.ceil(math.log2(max(node_count, 2))))
    raise SystemExit(f"Unsupported candidate mode: {mode}")


def _resolve_image_path(raw_path: Path) -> Path:
    if raw_path.exists():
        return raw_path
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / raw_path
    if candidate.exists():
        return candidate
    raise SystemExit(f"Image path not found: {raw_path}")


def _write_reports(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    json_path = output_dir / "rcm_scaling_report.json"
    csv_path = output_dir / "rcm_scaling_report.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "case",
        "kind",
        "nodes",
        "triangles",
        "candidate_mode",
        "max_candidates",
        "baseline_beta",
        "multistart_beta",
        "delta_beta",
        "baseline_time_ms",
        "multistart_time_ms",
        "time_overhead_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
