import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.geometry.geometry_utils import mesh_quality_report
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    images_dir = project_root / "data" / "images"
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_names = ("star.png", "u_shape.png", "hourglass.png", "one.png")
    h_reference = _estimate_reference_h(images_dir / "circle.png")
    if h_reference is None:
        h_reference = 20.0

    h_factors = (1.4, 1.0, 0.8, 0.6)
    rows: list[dict[str, object]] = []
    tracemalloc.start()

    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"[skip] missing image: {image_path}")
            continue

        preprocessed = preprocess_photo_to_boundary(image_path, threshold=127, epsilon=2.0)
        area = _polygon_area(preprocessed.boundary)

        for factor in h_factors:
            h = h_reference * factor
            mesher = AdvancingFrontMesher(
                min_triangle_quality=0.01,
                target_edge_length=h,
                smoothing_iterations=8,
            )
            tracemalloc.reset_peak()
            t0 = time.perf_counter()
            try:
                mesh = mesher.generate(preprocessed.boundary)
            except Exception as error:
                rows.append(
                    {
                        "image": image_name,
                        "h_factor": factor,
                        "target_h": h,
                        "status": f"failed: {error}",
                    }
                )
                continue
            t1 = time.perf_counter()
            peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
            quality = mesh_quality_report(mesh, bins=10)

            row = {
                "image": image_name,
                "h_factor": factor,
                "target_h": h,
                "status": "ok",
                "triangles": len(mesh.triangles),
                "area": area,
                "triangles_per_area": len(mesh.triangles) / area if area > 0 else 0.0,
                "quality_min": quality.min_quality,
                "quality_mean": quality.mean_quality,
                "quality_max": quality.max_quality,
                "time_total_ms": (t1 - t0) * 1000.0,
                "peak_mem_mb": peak_mem_mb,
            }
            rows.append(row)
            print(
                f"{image_name} h={h:.3f}: triangles={row['triangles']}, "
                f"q_mean={row['quality_mean']:.3f}, time_ms={row['time_total_ms']:.1f}, mem_mb={peak_mem_mb:.2f}"
            )

    tracemalloc.stop()
    _write_benchmark_reports(output_dir, rows)


def _estimate_reference_h(reference_image: Path) -> float | None:
    if not reference_image.exists():
        return None
    preprocessed = preprocess_photo_to_boundary(reference_image, threshold=127, epsilon=2.0)
    boundary = preprocessed.boundary
    if len(boundary) < 2:
        return None
    perimeter = 0.0
    for i in range(len(boundary)):
        a = boundary[i]
        b = boundary[(i + 1) % len(boundary)]
        perimeter += ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
    return perimeter / len(boundary)


def _polygon_area(polygon: list) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        area += p1.x * p2.y - p2.x * p1.y
    return abs(area) / 2.0


def _write_benchmark_reports(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    json_path = output_dir / "stress_benchmark_report.json"
    csv_path = output_dir / "stress_benchmark_report.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "image",
        "h_factor",
        "target_h",
        "status",
        "triangles",
        "area",
        "triangles_per_area",
        "quality_min",
        "quality_mean",
        "quality_max",
        "time_total_ms",
        "peak_mem_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
