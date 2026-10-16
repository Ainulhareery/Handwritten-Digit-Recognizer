"""Regression tests from actual photos. Run alongside the existing test suite.

These tests validate segmentation regions, NOT CNN classification accuracy.
The photos were used during debugging and are not held-out evaluation data.
"""

import unittest
from pathlib import Path

from PIL import Image

from src.segmentation import segment_digits


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class NewPhotoSegmentationTests(unittest.TestCase):
    def _check_segments(self, filename, expected_count):
        image = Image.open(FIXTURES / filename)
        segments = segment_digits(image)
        self.assertEqual(len(segments), expected_count)
        self.assertEqual(
            [region.box[0] for region in segments],
            sorted(region.box[0] for region in segments),
        )
        for region in segments:
            left, top, right, bottom = region.box
            self.assertTrue(0 <= left < right <= image.width)
            self.assertTrue(0 <= top < bottom <= image.height)
            self.assertGreater(region.image.width, 0)
            self.assertGreater(region.image.height, 0)
        return segments

    def test_handwritten_347_has_three_regions(self):
        regions = self._check_segments("real_347.jpg", 3)
        # Ensure the first region includes the main lower stroke of the 3.
        self.assertGreater(regions[0].box[3] - regions[0].box[1], 150)

    def test_handwritten_962_has_three_regions(self):
        self._check_segments("real_962.jpg", 3)

    def test_another_handwritten_three_digit_photo(self):
        self._check_segments("real_three_digits_b.jpg", 3)


if __name__ == "__main__":
    unittest.main()
