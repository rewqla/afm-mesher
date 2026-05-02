import sys
import csv
import json
import time
import tracemalloc
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.geometry.geometry_utils import mesh_average_quality, mesh_quality_report
from src.infrastructure.export.mesh_exporter import (
    export_mesh_obj,
    export_mesh_ply,
    export_mesh_vtk,
)
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary
from src.infrastructure.visualization.debug_visualizer import save_debug_visualization


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    images_dir = project_root / "data" / "images"
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = (
        "circle.png",
        "square.png",
        "l_shape.png",
        "star.png",
        "u_shape.png",
        "hourglass.png",
        "one.png",
        "spot_with_circle.png",
        "spot_with_lines.png",
    )
    user_h: float | None = None
    default_h = 20.0
    reference_image = images_dir / "circle.png"

    auto_h = _estimate_reference_h(reference_image)
    target_h = user_h if user_h is not None else (auto_h if auto_h is not None else default_h)
    print(f"[info] target edge length h = {target_h:.3f}")

    mesher = AdvancingFrontMesher(
        min_triangle_quality=0.01,
        target_edge_length=target_h,
        smoothing_iterations=8,
    )
    report_rows: list[dict[str, object]] = []
    tracemalloc.start()

    for image_name in test_images:
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"[skip] missing: {image_path}")
            continue
        tracemalloc.reset_peak()
        t0 = time.perf_counter()

        try:
            preprocessed = preprocess_photo_to_boundary(
                image_path=image_path,
                threshold=127,
                epsilon=2.0,
            )
        except Exception as error:
            print(f"[skip] preprocessing failed for {image_name}: {error}")
            continue
        t1 = time.perf_counter()

        try:
            mesh = mesher.generate(preprocessed.boundary)
        except ValueError as error:
            print(f"[skip] meshing failed for {image_name}: {error}")
            continue
        t2 = time.perf_counter()

        output_path = output_dir / f"{image_path.stem}_debug.png"
        try:
            save_debug_visualization(
                image_path=image_path,
                boundary=preprocessed.boundary,
                mesh=mesh,
                output_path=output_path,
            )
        except Exception as error:
            print(f"[skip] visualization failed for {image_name}: {error}")
            continue

        obj_path = output_dir / f"{image_path.stem}.obj"
        ply_path = output_dir / f"{image_path.stem}.ply"
        vtk_path = output_dir / f"{image_path.stem}.vtk"
        export_mesh_obj(mesh, obj_path)
        export_mesh_ply(mesh, ply_path)
        export_mesh_vtk(mesh, vtk_path)
        t3 = time.perf_counter()

        quality = mesh_quality_report(mesh, bins=10)
        avg_quality = quality.mean_quality
        peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
        polygon_area = _polygon_area(preprocessed.boundary)
        triangles_per_area = len(mesh.triangles) / polygon_area if polygon_area > 0 else 0.0

        print(
            f"{image_name}: boundary_points={len(preprocessed.boundary)}, "
            f"triangles={len(mesh.triangles)}, "
            f"quality(min/mean/max)=({quality.min_quality:.3f}/{quality.mean_quality:.3f}/{quality.max_quality:.3f}), "
            f"time_total_ms={(t3 - t0) * 1000:.1f}, peak_mem_mb={peak_mem_mb:.2f}, "
            f"output={output_path}, obj={obj_path.name}, ply={ply_path.name}, vtk={vtk_path.name}"
        )
        if avg_quality < 0.7:
            print(f"[warn] {image_name}: average quality below target (0.7)")

        report_rows.append(
            {
                "image": image_name,
                "boundary_points": len(preprocessed.boundary),
                "triangles": len(mesh.triangles),
                "area": polygon_area,
                "triangles_per_area": triangles_per_area,
                "quality_min": quality.min_quality,
                "quality_mean": quality.mean_quality,
                "quality_max": quality.max_quality,
                "quality_hist_bins": quality.histogram_bins,
                "quality_hist_counts": quality.histogram_counts,
                "time_preprocess_ms": (t1 - t0) * 1000.0,
                "time_meshing_ms": (t2 - t1) * 1000.0,
                "time_export_ms": (t3 - t2) * 1000.0,
                "time_total_ms": (t3 - t0) * 1000.0,
                "peak_mem_mb": peak_mem_mb,
            }
        )
    tracemalloc.stop()
    _write_reports(output_dir, report_rows)


def _estimate_reference_h(reference_image: Path) -> float | None:
    if not reference_image.exists():
        return None
    preprocessed = preprocess_photo_to_boundary(
        image_path=reference_image,
        threshold=127,
        epsilon=2.0,
    )
    boundary = preprocessed.boundary
    if len(boundary) < 2:
        return None

    perimeter = 0.0
    for i in range(len(boundary)):
        a = boundary[i]
        b = boundary[(i + 1) % len(boundary)]
        dx = b.x - a.x
        dy = b.y - a.y
        perimeter += (dx * dx + dy * dy) ** 0.5
    return perimeter / len(boundary)


def _polygon_area(polygon: list) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        area += p1.x * p2.y - p2.x * p1.y
    return abs(area) / 2.0


def _write_reports(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    json_path = output_dir / "mesh_quality_report.json"
    csv_path = output_dir / "mesh_benchmark_report.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "image",
        "boundary_points",
        "triangles",
        "area",
        "triangles_per_area",
        "quality_min",
        "quality_mean",
        "quality_max",
        "quality_hist_bins",
        "quality_hist_counts",
        "time_preprocess_ms",
        "time_meshing_ms",
        "time_export_ms",
        "time_total_ms",
        "peak_mem_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["quality_hist_bins"] = json.dumps(out_row["quality_hist_bins"])
            out_row["quality_hist_counts"] = json.dumps(out_row["quality_hist_counts"])
            writer.writerow(out_row)


if __name__ == "__main__":
    main()
