from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from _image_assets import ensure_basic_test_images
from src.application.services.advancing_front_mesher import AdvancingFrontMesher
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary
from src.infrastructure.visualization.debug_visualizer import save_debug_visualization


class TestMainPipeline(unittest.TestCase):
    def test_pipeline_from_image_to_debug_output(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        images_dir = tests_dir / "images"
        output_dir = tests_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_basic_test_images(images_dir)

        image_path = images_dir / "square.png"
        output_path = output_dir / "test_square_debug.png"

        preprocessed = preprocess_photo_to_boundary(image_path, threshold=127, epsilon=2.0)
        mesh = AdvancingFrontMesher(min_triangle_quality=0.01).generate(preprocessed.boundary)
        save_debug_visualization(image_path, preprocessed.boundary, mesh, output_path)

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
