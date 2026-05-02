from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from _image_assets import ensure_complex_test_images
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary
from src.infrastructure.visualization.debug_visualizer import save_debug_visualization


class TestComplexPipelineOutputs(unittest.TestCase):
    def test_pipeline_outputs_for_complex_images(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        images_dir = tests_dir / "images"
        output_dir = tests_dir / "output"
        ensure_complex_test_images(images_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mesher = AdvancingFrontMesher(min_triangle_quality=0.01, smoothing_iterations=8)
        image_names = ("star.png", "u_shape.png", "hourglass.png")

        for image_name in image_names:
            with self.subTest(image=image_name):
                image_path = images_dir / image_name

                preprocessed = preprocess_photo_to_boundary(image_path, threshold=127, epsilon=2.0)
                mesh = mesher.generate(preprocessed.boundary)
                output_path = output_dir / f"test_{image_path.stem}_debug.png"
                save_debug_visualization(
                    image_path=image_path,
                    boundary=preprocessed.boundary,
                    mesh=mesh,
                    output_path=output_path,
                )

                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
