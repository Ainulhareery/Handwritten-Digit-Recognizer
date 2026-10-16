"""Run as: python -m unittest discover -s tests -p 'test_multi_digit.py' -v"""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from src.segmentation import segment_digits
from src.multi_digit import recognize_number


def make_digits(*, inverted=False, digits=3):
    bg, ink = (0, 255) if inverted else (255, 0)
    image = Image.new("RGB", (300, 100), (bg, bg, bg))
    drawing = ImageDraw.Draw(image)
    for idx in range(digits):
        x = 20 + idx * 92
        # Distinct stroked shapes, spaced apart, not font-generated claims.
        if idx % 3 == 0:
            drawing.line([(x, 20), (x + 45, 20), (x, 75), (x + 45, 75)], fill=(ink,)*3, width=6)
        elif idx % 3 == 1:
            drawing.line([(x + 10, 70), (x + 35, 18), (x + 35, 78)], fill=(ink,)*3, width=6)
            drawing.line([(x, 52), (x + 47, 52)], fill=(ink,)*3, width=6)
        else:
            drawing.line([(x, 21), (x + 45, 21), (x + 16, 78)], fill=(ink,)*3, width=6)
    return image


class SegmentationTests(unittest.TestCase):
    def test_separated_dark_digits(self):
        regions = segment_digits(make_digits())
        self.assertEqual(len(regions), 3)
        self.assertEqual(sorted(item.box[0] for item in regions), [item.box[0] for item in regions])

    def test_separated_light_digits(self):
        self.assertEqual(len(segment_digits(make_digits(inverted=True))), 3)

    def test_blank_image_rejected(self):
        with self.assertRaises(ValueError):
            segment_digits(Image.new("RGB", (300, 100), "white"))

    def test_single_digit_rejected(self):
        with self.assertRaises(ValueError):
            segment_digits(make_digits(digits=1))

    def test_pipeline_with_mock_classifier(self):
        calls = []
        predicted = iter([2, 4, 7])

        def fake_predict(model_input):
            self.assertEqual(model_input.shape, (1, 28, 28, 1))
            self.assertTrue(np.isfinite(model_input).all())
            self.assertGreaterEqual(model_input.min(), 0)
            self.assertLessEqual(model_input.max(), 1)
            digit = next(predicted)
            calls.append(digit)
            scores = np.zeros(10, dtype=np.float32)
            scores[digit] = 1.0
            return digit, scores

        number, details = recognize_number(make_digits(), predict_fn=fake_predict)
        self.assertEqual(number, "247")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(details), 3)


if __name__ == "__main__":
    unittest.main()
