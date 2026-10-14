"""Extract clearly separated handwritten digits, left to right.

This is a deliberately constrained first version: digits must occupy
separate horizontal (x-axis) ranges. Touching/overlapping digits, paper
rules, and complex photos are not supported reliably.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DigitSegment:
    """A digit crop and its bounding box (left, top, right, bottom)."""

    image: Image.Image
    box: tuple[int, int, int, int]


def _as_grayscale(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    # Canvas output may have an alpha channel. Composite over white first.
    if image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        rgb = rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

    gray = image.convert("L")
    gray.thumbnail((1200, 800), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.uint8)


def segment_digits(image: Image.Image, *, max_digits: int = 5) -> list[DigitSegment]:
    """Find 2 to max_digits non-overlapping digit regions.

    Raises ValueError rather than silently dropping detected segments.
    The returned crops preserve ink polarity (black or white strokes).
    """
    if not 2 <= max_digits <= 20:
        raise ValueError("max_digits must be between 2 and 20.")

    gray = _as_grayscale(image)
    height, width = gray.shape
    if height < 12 or width < 12:
        raise ValueError("Image is too small for multi-digit recognition.")

    # Use the predominant background brightness to select ink polarity.
    dark_ink = float(np.median(gray)) >= 128.0
    threshold_mode = cv2.THRESH_BINARY_INV if dark_ink else cv2.THRESH_BINARY
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(smoothed, 0, 255, threshold_mode | cv2.THRESH_OTSU)

    # Reject small isolated speckles before looking for digit columns.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    min_area = max(4, int(height * width * 0.00003))
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if area < min_area:
            continue
        # Long, thin horizontal rules are more likely background artifacts.
        if w > 0.75 * width and h < 0.04 * height:
            continue
        cleaned[labels == component] = 255

    active_columns = np.flatnonzero(np.any(cleaned > 0, axis=0))
    if active_columns.size == 0:
        raise ValueError("No handwriting detected. Use a plain, clear background.")

    # Empty x-columns delimit individual digits. This intentionally does not
    # attempt to separate touching digits or overlapping x-ranges.
    breaks = np.flatnonzero(np.diff(active_columns) > 1)
    runs = np.split(active_columns, breaks + 1)
    if not 2 <= len(runs) <= max_digits:
        raise ValueError(
            f"Detected {len(runs)} digit region(s); expected 2–{max_digits}. "
            "Draw separate digits with visible gaps and try again."
        )

    background = 255 if dark_ink else 0
    regions = []
    for run in runs:
        left, right = int(run[0]), int(run[-1]) + 1
        ys, xs = np.where(cleaned[:, left:right] > 0)
        if ys.size == 0:
            continue
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left = left + int(xs.min())
        right = int(run[0]) + int(xs.max()) + 1
        region = gray[top:bottom, left:right]

        # Keep generous padding, letting the existing per-digit preprocessing
        # detect the ink and produce a 28x28 MNIST-style input.
        pad = max(5, round(0.25 * max(region.shape)))
        canvas = np.full(
            (region.shape[0] + 2 * pad, region.shape[1] + 2 * pad),
            background,
            dtype=np.uint8,
        )
        canvas[pad:pad + region.shape[0], pad:pad + region.shape[1]] = region
        regions.append(DigitSegment(Image.fromarray(canvas), (left, top, right, bottom)))

    return regions
