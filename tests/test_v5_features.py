"""Regression coverage for canvas, output boxes, and optional photo correction.

Screenshot-cropped canvas fixtures are UI reproductions, not held-out
handwriting datasets and not evidence of CNN prediction accuracy.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.image_quality import enhance_photo
from src.multi_digit import recognize_number
from src.segmentation import segment_digits
from src.visualization import annotate_regions

FIXTURES = Path(__file__).parent / "fixtures"


class CanvasSegmentationTests(unittest.TestCase):
    def test_all_three_canvas_cases_detect_three_regions(self):
        for sample in ("113259", "113333", "113316"):
            with self.subTest(sample=sample):
                image = Image.open(FIXTURES / f"canvas_from_screenshot_{sample}.png")
                segments = segment_digits(image)
                self.assertEqual(len(segments), 3)
                self.assertEqual(
                    [item.box[0] for item in segments],
                    sorted(item.box[0] for item in segments),
                )

    def test_transparent_canvas_composited_over_white(self):
        image = Image.open(FIXTURES / "canvas_from_screenshot_113259.png")
        rgba = image.convert("RGBA")
        self.assertEqual(len(segment_digits(rgba)), 3)

    def test_canvas_blank_image_rejected(self):
        with self.assertRaises(ValueError):
            segment_digits(Image.new("RGB", (300, 120), "white"))


class ImageAndDisplayTests(unittest.TestCase):
    def test_visualization_uses_original_coordinates(self):
        image = Image.new("RGB", (1500, 700), "white")
        details = [{"box": (570, 230, 660, 350)}]
        result = annotate_regions(image, details)
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((570, 240)), (255, 0, 0))
        self.assertEqual(result.getpixel((1200, 600)), (255, 255, 255))

    def test_visualization_rejects_bad_boxes(self):
        with self.assertRaises(ValueError):
            annotate_regions(Image.new("RGB", (100, 100)), [{"box": (20, 20, 120, 40)}])

    def test_photo_enhancement_keeps_image_size(self):
        image = Image.open(FIXTURES / "20261015_105347.jpg")
        self.assertEqual(enhance_photo(image).size, image.size)

    def test_existing_photo_347_still_has_three_regions(self):
        image = Image.open(FIXTURES / "20261015_105354.jpg")
        self.assertEqual(len(segment_digits(image)), 3)

    def test_existing_photo_9946_still_has_four_regions(self):
        image = Image.open(FIXTURES / "20261015_024707.jpg")
        self.assertEqual(len(segment_digits(image)), 4)

    @patch("src.multi_digit.segment_digits")
    def test_expected_count_forwarded_without_setting_predicted_digits(self, mock_segment):
        mock_segment.return_value = segment_digits(
            Image.open(FIXTURES / "canvas_from_screenshot_113259.png")
        )
        outputs = iter([2, 4, 7])

        def fake_predict(array):
            self.assertEqual(array.shape, (1, 28, 28, 1))
            digit = next(outputs)
            scores = np.zeros(10, dtype=np.float32)
            scores[digit] = 1.0
            return digit, scores

        image = Image.open(FIXTURES / "canvas_from_screenshot_113259.png")
        number, details = recognize_number(
            image, predict_fn=fake_predict, max_digits=5, expected_digits=3
        )
        self.assertEqual(number, "247")
        self.assertEqual(len(details), 3)
        self.assertEqual(mock_segment.call_args.kwargs["expected_digits"], 3)


if __name__ == "__main__":
    unittest.main()
