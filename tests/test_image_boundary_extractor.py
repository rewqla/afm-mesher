from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from _image_assets import ensure_basic_test_images
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary


class TestImageBoundaryExtractor(unittest.TestCase):
    def test_preprocess_photo_to_boundary(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        images_dir = tests_dir / "images"
        ensure_basic_test_images(images_dir)
        image_names = ("circle.png", "square.png", "l_shape.png")

        for image_name in image_names:
            with self.subTest(image_name=image_name):
                result = preprocess_photo_to_boundary(
                    images_dir / image_name,
                    threshold=127,
                    epsilon=2.0,
                )
                self.assertGreaterEqual(len(result.external_contour), 4)
                self.assertGreaterEqual(len(result.simplified_contour), 3)
                self.assertGreaterEqual(len(result.boundary), 3)
                self.assertLessEqual(len(result.simplified_contour), len(result.external_contour))


if __name__ == "__main__":
    unittest.main()
