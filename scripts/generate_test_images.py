from pathlib import Path
import sys

from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.dto.complex_test_contours import (
    hourglass_contour,
    star_contour,
    u_shape_contour,
)


def generate_test_images(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _generate_circle(output_dir / "circle.png")
    _generate_square(output_dir / "square.png")
    _generate_l_shape(output_dir / "l_shape.png")
    _generate_complex_contour(output_dir / "star.png", star_contour())
    _generate_complex_contour(output_dir / "u_shape.png", u_shape_contour())
    _generate_complex_contour(output_dir / "hourglass.png", hourglass_contour())


def _generate_circle(path: Path) -> None:
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 216, 216), fill=255)
    image.save(path)


def _generate_square(path: Path) -> None:
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 208, 208), fill=255)
    image.save(path)


def _generate_l_shape(path: Path) -> None:
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
    # Concave L-shape for contour extraction tests.
    draw.polygon(
        (
            (48, 48),
            (208, 48),
            (208, 104),
            (112, 104),
            (112, 208),
            (48, 208),
        ),
        fill=255,
    )
    image.save(path)


def _generate_complex_contour(path: Path, contour) -> None:
    image = Image.new("L", (500, 500), color=0)
    draw = ImageDraw.Draw(image)
    polygon = [(point.x, point.y) for point in contour]
    draw.polygon(polygon, fill=255)
    image.save(path)


if __name__ == "__main__":
    generate_test_images(Path("data/images"))
