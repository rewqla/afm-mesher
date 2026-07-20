from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities.indexed_mesh import IndexedMesh
from src.domain.entities.mesh import Mesh
from src.domain.entities.node_type import NodeType
from src.domain.geometry.geometry_utils import mesh_average_quality, mesh_quality_report

_KEY_PRECISION = 10


_MISSING_SOURCE_CONTOURS_MESSAGE = (
    "Координати вершин доступні лише для областей, заданих через імпорт координат. "
    "Ця сітка побудована іншим способом."
)


def export_coordinates(mesh: Mesh, path: Path) -> None:
    if mesh.source_contours is None:
        raise ValueError(_MISSING_SOURCE_CONTOURS_MESSAGE)

    source = mesh.source_contours
    payload = {
        "outer_boundary": [[x, y] for x, y in source.outer_boundary],
        "inclusions": [
            {
                "vertices": [[x, y] for x, y in vertices],
                "type": inclusion_type,
            }
            for vertices, inclusion_type in source.inclusions
        ],
        "cuts": [
            [[x, y] for x, y in cut]
            for cut in source.cuts
        ],
    }
    _write_json(path, payload)


def export_triangles(mesh: Mesh, path: Path) -> None:
    payload = {
        "triangles": [
            {
                "triangle_number": triangle.triangle_number,
                "degree": "linear",
                "node_numbers": list(triangle.node_numbers),
                "node_coordinates": [[point.x, point.y] for point in triangle.node_coordinates],
                "area": triangle.area,
            }
            for triangle in mesh.linear_triangles(key_precision=_KEY_PRECISION)
        ],
        "meters_per_pixel": mesh.meters_per_pixel,
    }
    _write_json(path, payload)


def export_nodes(indexed_mesh: IndexedMesh, path: Path) -> None:
    payload = {
        "nodes": [
            {
                "node_number": node_id,
                "coordinates": [x, y],
                "attribute": _node_type_label(indexed_mesh.node_type(node_id)),
            }
            for node_id, (x, y) in _iter_indexed_nodes(indexed_mesh)
        ],
        "index_base": indexed_mesh.index_base,
        "meters_per_pixel": indexed_mesh.meters_per_pixel,
    }
    _write_json(path, payload)


def export_statistics(mesh: Mesh, path: Path, bins: int = 10) -> None:
    triangle_count = len(mesh.triangles)
    indexed_mesh = mesh.indexed_mesh_data
    if indexed_mesh is None and triangle_count > 0:
        indexed_mesh = mesh.indexed_mesh(key_precision=_KEY_PRECISION)

    node_count = len(indexed_mesh.nodes) if indexed_mesh is not None else 0
    bandwidth = indexed_mesh.bandwidth if indexed_mesh is not None else 0

    if triangle_count > 0:
        quality_report = mesh_quality_report(mesh, bins=bins)
        quality_mean = mesh_average_quality(mesh)
        percentages = [
            round(count / triangle_count * 100.0, 2)
            for count in quality_report.histogram_counts
        ]
        quality = {
            "min": round(quality_report.min_quality, 2),
            "mean": round(quality_mean, 2),
            "max": round(quality_report.max_quality, 2),
        }
        quality_histogram = {
            "bins": [round(value, 2) for value in quality_report.histogram_bins],
            "counts": quality_report.histogram_counts,
            "percentages": percentages,
        }
    else:
        quality = {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }
        quality_histogram = {
            "bins": [round(i / bins, 2) for i in range(bins + 1)] if bins >= 1 else [],
            "counts": [0 for _ in range(bins)] if bins >= 1 else [],
            "percentages": [0.0 for _ in range(bins)] if bins >= 1 else [],
        }

    payload = {
        "triangle_count": triangle_count,
        "node_count": node_count,
        "bandwidth": bandwidth,
        "quality": quality,
        "quality_histogram": quality_histogram,
    }
    _write_json(path, payload)


def _iter_indexed_nodes(indexed_mesh: IndexedMesh) -> list[tuple[int, tuple[float, float]]]:
    return [
        (indexed_mesh.index_base + offset, coordinates)
        for offset, coordinates in enumerate(indexed_mesh.nodes)
    ]


def _node_type_label(node_type: NodeType) -> str:
    if node_type == NodeType.INTERFACE:
        return "cut"
    return node_type.value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
