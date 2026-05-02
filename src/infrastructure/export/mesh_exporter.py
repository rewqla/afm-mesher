from __future__ import annotations

from pathlib import Path

from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point

_KEY_PRECISION = 10


def export_mesh_obj(mesh: Mesh, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = _mesh_vertices_faces(mesh)

    lines: list[str] = ["# AFM mesh OBJ export"]
    lines.extend(f"v {p.x:.10f} {p.y:.10f} 0.0" for p in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_mesh_ply(mesh: Mesh, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = _mesh_vertices_faces(mesh)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertex_lines = [f"{p.x:.10f} {p.y:.10f} 0.0" for p in vertices]
    face_lines = [f"3 {a} {b} {c}" for a, b, c in faces]

    output_path.write_text("\n".join(header + vertex_lines + face_lines) + "\n", encoding="utf-8")
    return output_path


def export_mesh_vtk(mesh: Mesh, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = _mesh_vertices_faces(mesh)

    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "AFM mesh VTK export",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(vertices)} float",
    ]
    lines.extend(f"{p.x:.10f} {p.y:.10f} 0.0" for p in vertices)
    lines.append(f"POLYGONS {len(faces)} {len(faces) * 4}")
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _mesh_vertices_faces(mesh: Mesh) -> tuple[list[Point], list[tuple[int, int, int]]]:
    key_to_index: dict[tuple[float, float], int] = {}
    vertices: list[Point] = []
    faces: list[tuple[int, int, int]] = []

    for triangle in mesh.triangles:
        idx_a = _index_for_point(triangle.a, key_to_index, vertices)
        idx_b = _index_for_point(triangle.b, key_to_index, vertices)
        idx_c = _index_for_point(triangle.c, key_to_index, vertices)
        faces.append((idx_a, idx_b, idx_c))

    return vertices, faces


def _index_for_point(
    point: Point,
    key_to_index: dict[tuple[float, float], int],
    vertices: list[Point],
) -> int:
    key = (round(point.x, _KEY_PRECISION), round(point.y, _KEY_PRECISION))
    idx = key_to_index.get(key)
    if idx is not None:
        return idx
    idx = len(vertices)
    key_to_index[key] = idx
    vertices.append(point)
    return idx
