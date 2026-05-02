from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from _image_assets import ensure_complex_test_images
from src.infrastructure.image.photo_preprocessor import preprocess_photo_to_boundary


class TestPreprocessingComplexImages(unittest.TestCase):
    def test_preprocessing_for_complex_shapes(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        images_dir = tests_dir / "images"
        ensure_complex_test_images(images_dir)
        image_names = ("star.png", "u_shape.png", "hourglass.png")

        for image_name in image_names:
            with self.subTest(image=image_name):
                image_path = images_dir / image_name

                result = preprocess_photo_to_boundary(
                    image_path=image_path,
                    threshold=127,
                    epsilon=2.0,
                )

                self.assertGreaterEqual(len(result.external_contour), 8)
                self.assertGreaterEqual(len(result.simplified_contour), 3)
                self.assertGreaterEqual(len(result.boundary), 3)
                self.assertLessEqual(len(result.simplified_contour), len(result.external_contour))


if __name__ == "__main__":
    unittest.main()
