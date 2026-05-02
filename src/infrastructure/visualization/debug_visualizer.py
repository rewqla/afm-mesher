from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.domain.entities.mesh import Mesh
from src.domain.entities.point import Point


def save_debug_visualization(
    image_path: str | Path,
    boundary: list[Point],
    mesh: Mesh,
    output_path: str | Path,
) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(image, cmap="gray")

    # Draw generated mesh triangles.
    for triangle in mesh.triangles:
        xs = [triangle.a.x, triangle.b.x, triangle.c.x, triangle.a.x]
        ys = [triangle.a.y, triangle.b.y, triangle.c.y, triangle.a.y]
        ax.plot(xs, ys, color="deepskyblue", linewidth=0.6, alpha=0.85)

    # Draw simplified boundary on top for easier debugging.
    if boundary:
        boundary_x = [point.x for point in boundary] + [boundary[0].x]
        boundary_y = [point.y for point in boundary] + [boundary[0].y]
        ax.plot(boundary_x, boundary_y, color="crimson", linewidth=1.8)

    ax.set_axis_off()
    ax.set_title("Boundary + AFM Mesh", fontsize=10)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
