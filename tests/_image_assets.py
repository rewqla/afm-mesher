from pathlib import Path

from PIL import Image, ImageDraw

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import tests._bootstrap  # type: ignore  # noqa: F401
from src.application.dto.complex_test_contours import (
    hourglass_contour,
    star_contour,
    u_shape_contour,
)


def ensure_basic_test_images(images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    _save_circle(images_dir / "circle.png")
    _save_square(images_dir / "square.png")
    _save_l_shape(images_dir / "l_shape.png")


def ensure_complex_test_images(images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    _save_polygon(images_dir / "star.png", star_contour())
    _save_polygon(images_dir / "u_shape.png", u_shape_contour())
    _save_polygon(images_dir / "hourglass.png", hourglass_contour())


def _save_circle(path: Path) -> None:
    if path.exists():
        return
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 216, 216), fill=255)
    image.save(path)


def _save_square(path: Path) -> None:
    if path.exists():
        return
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 208, 208), fill=255)
    image.save(path)


def _save_l_shape(path: Path) -> None:
    if path.exists():
        return
    image = Image.new("L", (256, 256), color=0)
    draw = ImageDraw.Draw(image)
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


def _save_polygon(path: Path, contour) -> None:
    if path.exists():
        return
    image = Image.new("L", (500, 500), color=0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(p.x, p.y) for p in contour], fill=255)
    image.save(path)
