from pathlib import Path
import unittest

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import tests._bootstrap  # type: ignore  # noqa: F401

try:
    from _image_assets import ensure_basic_test_images, ensure_complex_test_images
except ModuleNotFoundError:
    from tests._image_assets import ensure_basic_test_images, ensure_complex_test_images
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.domain.geometry.geometry_utils import mesh_average_quality
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary


class TestImageQualityThresholds(unittest.TestCase):
    def test_avg_quality_threshold_for_input_images(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        images_dir = tests_dir / "images"
        ensure_basic_test_images(images_dir)
        ensure_complex_test_images(images_dir)

        # Keep mesher configuration consistent with main.py workflow.
        target_h = self._estimate_reference_h(images_dir / "circle.png")
        mesher = AdvancingFrontMesher(
            min_triangle_quality=0.01,
            target_edge_length=target_h,
            smoothing_iterations=8,
        )

        image_names = (
            "circle.png",
            "square.png",
            "l_shape.png",
            "star.png",
            "u_shape.png",
            "hourglass.png",
        )

        for image_name in image_names:
            with self.subTest(image=image_name):
                image_path = images_dir / image_name
                preprocessed = preprocess_photo_to_boundary(
                    image_path=image_path,
                    threshold=127,
                    epsilon=2.0,
                )
                mesh = mesher.generate(preprocessed.boundary)
                avg_quality = mesh_average_quality(mesh)
                self.assertGreaterEqual(
                    avg_quality,
                    0.7,
                    msg=f"{image_name} avg_quality={avg_quality:.3f} < 0.7",
                )

    def _estimate_reference_h(self, reference_image: Path) -> float:
        preprocessed = preprocess_photo_to_boundary(
            image_path=reference_image,
            threshold=127,
            epsilon=2.0,
        )
        boundary = preprocessed.boundary
        perimeter = 0.0
        for i in range(len(boundary)):
            a = boundary[i]
            b = boundary[(i + 1) % len(boundary)]
            dx = b.x - a.x
            dy = b.y - a.y
            perimeter += (dx * dx + dy * dy) ** 0.5
        return perimeter / len(boundary)


if __name__ == "__main__":
    unittest.main()
