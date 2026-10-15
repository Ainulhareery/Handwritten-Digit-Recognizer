"""Run: python -m unittest discover -s tests -p 'test_*.py' -v"""

import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from src.segmentation import segment_digits


class SegmentationRegressionTests(unittest.TestCase):
    def test_difficult_notebook_photo_four_regions(self):
        path = Path(__file__).parent / "fixtures" / "difficult_5729.jpg"
        image = Image.open(path)
        regions = segment_digits(image)
        self.assertEqual(len(regions), 4)
        self.assertEqual([s.box[0] for s in regions], sorted(s.box[0] for s in regions))
        for segment in regions:
            left, top, right, bottom = segment.box
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, image.width)
            self.assertLessEqual(bottom, image.height)

    def test_difficult_photo_expected_digit_count(self):
        path = Path(__file__).parent / "fixtures" / "difficult_5729.jpg"
        regions = segment_digits(Image.open(path), expected_digits=4)
        self.assertEqual(len(regions), 4)

    def test_rejects_invalid_expected_count(self):
        with self.assertRaises(ValueError):
            segment_digits(Image.new("RGB", (400, 130), "white"), expected_digits=7)

    def test_preprocessed_regions_for_cnn(self):
        from src.preprocessing import preprocess_image

        path = Path(__file__).parent / "fixtures" / "difficult_5729.jpg"
        for segment in segment_digits(Image.open(path)):
            _, batch = preprocess_image(segment.image)
            self.assertEqual(batch.shape, (1, 28, 28, 1))
            self.assertGreater(float(batch.max()), 0.0)
            self.assertLessEqual(float(batch.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
