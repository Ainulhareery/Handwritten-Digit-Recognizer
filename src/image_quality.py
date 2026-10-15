"""Optional, conservative enhancement for low-contrast handwritten photographs.

This does not recover information missing from a severely out-of-focus photo.
The output size is deliberately unchanged so detection boxes stay aligned.
"""

import cv2
import numpy as np
from PIL import Image


def enhance_photo(image: Image.Image) -> Image.Image:
    """Improve local contrast and lightly sharpen an RGB photo."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    boosted = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    smoothed = cv2.GaussianBlur(boosted, (0, 0), sigmaX=1.2)
    enhanced = cv2.addWeighted(boosted, 1.25, smoothed, -0.25, 0)
    return Image.fromarray(enhanced)
