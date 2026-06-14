from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from PySide6.QtGui import QImage

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
except ImportError as exc:  # pragma: no cover - diagnostic script
    raise SystemExit(
        "scipy is required for this diagnostic script. Install it in the active environment first."
    ) from exc

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.services import advancing_front_mesher as afm_module
from src.application.services.mesh_postprocessing import _build_node_adjacency, _normalize_triangle_indices, compute_max_difference
from src.domain.entities.linear_triangle import LinearTriangle
from src.domain.entities.point import Point
from src.presentation.tools import TriangulationMode
from src.presentation.triangulation_adapter import TriangulationAdapter, TriangulationSettings


def main() -> None:
    args = _parse_args()
    image_path = _resolve_image_path(args.image)

    image = QImage(str(image_path))
    if image.isNull():
        raise SystemExit(f"Failed to load image: {image_path}")

    adapter = TriangulationAdapter(threshold=args.threshold)
    settings = _resolve_settings(adapter, args)
    captured: dict[str, object] = {}

    original_rcm = afm_module.renumber_nodes_rcm

    def _capturing_rcm(
        nodes: list[tuple[float, float]],
        triangles: list[tuple[int, int, int]],
    ) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]], int]:
        result = original_rcm(nodes, triangles)
        captured["nodes"] = list(nodes)
        captured["triangles"] = [tuple(triangle) for triangle in triangles]
        captured["result"] = result
        return result

    afm_module.renumber_nodes_rcm = _capturing_rcm
    try:
        mesh, coefficient = adapter.run(image, mode=TriangulationMode.CUSTOM, custom_settings=settings)
    finally:
        afm_module.renumber_nodes_rcm = original_rcm

    raw_nodes = captured.get("nodes")
    raw_triangles = captured.get("triangles")
    own_result = captured.get("result")
    if not isinstance(raw_nodes, list) or not isinstance(raw_triangles, list) or not isinstance(own_result, tuple):
        raise SystemExit("Failed to capture raw graph data from renumber_nodes_rcm().")

    raw_nodes_typed = [tuple(node) for node in raw_nodes]
    raw_triangles_typed = [tuple(triangle) for triangle in raw_triangles]

    normalized_triangles, index_base = _normalize_triangle_indices(raw_triangles_typed, len(raw_nodes_typed))
    adjacency = _build_node_adjacency(len(raw_nodes_typed), normalized_triangles)
    graph = _build_csr_adjacency(adjacency)
    scipy_perm = reverse_cuthill_mckee(graph, symmetric_mode=True)

    own_nodes, own_triangles, own_beta = own_result
    own_triangles_one_based = _to_one_based([tuple(triangle) for triangle in own_triangles], index_base)
    scipy_nodes, scipy_triangles = _apply_permutation(raw_nodes_typed, normalized_triangles, scipy_perm)

    own_beta_check = compute_max_difference(_build_linear_triangles(own_nodes, own_triangles_one_based))
    scipy_beta = compute_max_difference(_build_linear_triangles(scipy_nodes, scipy_triangles))

    degrees = [len(neighbors) for neighbors in adjacency]
    own_start = min(range(len(raw_nodes_typed)), key=lambda node_id: (degrees[node_id], node_id))
    raw_coord_to_id = {tuple(coord): node_id + index_base for node_id, coord in enumerate(raw_nodes_typed)}
    own_perm_ids = [raw_coord_to_id[tuple(coord)] for coord in own_nodes]
    scipy_perm_ids = [raw_coord_to_id[tuple(coord)] for coord in scipy_nodes]

    print(f"image={image_path}")
    print(f"mode=CUSTOM target_h={settings.target_edge_length} epsilon={settings.contour_epsilon}")
    print(f"raw_graph nodes={len(raw_nodes_typed)} triangles={len(raw_triangles_typed)} adjacency=list[set[int]]")
    print(f"own_start_node={own_start + index_base} own_start_degree={degrees[own_start]}")
    print(f"own_perm_first_node={own_perm_ids[0] if own_perm_ids else None}")
    print(f"scipy_perm_first_node={scipy_perm_ids[0] if scipy_perm_ids else None}")
    print(f"own_beta={own_beta} own_beta_via_compute_max_difference={own_beta_check}")
    print(f"scipy_beta={scipy_beta}")
    print(f"delta={own_beta - scipy_beta}")
    print(f"mesh_average_quality={coefficient}")
    print(f"mesh_indexed_bandwidth={mesh.indexed_mesh_data.bandwidth if mesh.indexed_mesh_data else None}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the project RCM implementation with SciPy RCM.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(Path("data") / "images" / "one.png"),
        help="Input image path used to reconstruct the mesh through the existing triangulation pipeline.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binary threshold passed to the triangulation adapter.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="custom",
        choices=[mode.value for mode in TriangulationMode],
        help="Triangulation mode.",
    )
    parser.add_argument("--target-edge-length", type=float, default=20.0)
    parser.add_argument("--smoothing-iterations", type=int, default=5)
    parser.add_argument("--contour-epsilon", type=float, default=2.0)
    parser.add_argument("--max-iterations-factor", type=int, default=260)
    parser.add_argument("--min-triangle-quality", type=float, default=0.01)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / raw_path
    if candidate.exists():
        return candidate
    raise SystemExit(f"Image path not found: {raw_path}")


def _resolve_settings(adapter: TriangulationAdapter, args: argparse.Namespace) -> TriangulationSettings:
    mode = TriangulationMode(args.mode)
    if mode == TriangulationMode.CUSTOM:
        return TriangulationSettings(
            target_edge_length=args.target_edge_length,
            smoothing_iterations=args.smoothing_iterations,
            contour_epsilon=args.contour_epsilon,
            max_iterations_factor=args.max_iterations_factor,
            min_triangle_quality=args.min_triangle_quality,
            meters_per_pixel=args.meters_per_pixel,
        )
    preset = adapter.preset_settings(mode)
    return replace(preset, meters_per_pixel=args.meters_per_pixel)


def _build_csr_adjacency(adjacency: list[set[int]]) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for row, neighbors in enumerate(adjacency):
        for col in neighbors:
            rows.append(row)
            cols.append(col)
            data.append(1)
    size = len(adjacency)
    return csr_matrix((data, (rows, cols)), shape=(size, size))


def _apply_permutation(
    nodes: list[tuple[float, float]],
    triangles_zero_based: list[tuple[int, int, int]],
    permutation: object,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    perm = [int(node_id) for node_id in permutation]  # type: ignore[arg-type]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(perm)}
    permuted_nodes = [nodes[old_id] for old_id in perm]
    remapped_triangles = [
        tuple(old_to_new[node_id] + 1 for node_id in triangle)
        for triangle in triangles_zero_based
    ]
    return permuted_nodes, remapped_triangles


def _build_linear_triangles(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int]],
) -> list[LinearTriangle]:
    points = [Point(*node) for node in nodes]
    linear_triangles: list[LinearTriangle] = []
    for triangle_number, triangle in enumerate(triangles, start=1):
        coordinates = tuple(points[node_id - 1] for node_id in triangle)
        linear_triangles.append(
            LinearTriangle(
                triangle_number=triangle_number,
                node_coordinates=coordinates,  # type: ignore[arg-type]
                node_numbers=triangle,
            )
        )
    return linear_triangles


def _to_one_based(triangles: list[tuple[int, int, int]], index_base: int) -> list[tuple[int, int, int]]:
    if index_base == 1:
        return triangles
    return [tuple(node_id + 1 for node_id in triangle) for triangle in triangles]


if __name__ == "__main__":
    main()
