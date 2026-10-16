"""Classify each segmented digit using the project's existing CNN."""

from collections.abc import Callable

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image
from src.segmentation import segment_digits


def recognize_number(
    image: Image.Image,
    *,
    predict_fn: Callable,
    max_digits: int = 5,
    expected_digits: int | None = None,
    canvas_mode: bool = False,
) -> tuple[str, list[dict]]:
    """Return the full number as *text* and diagnostic data for each digit.

    Retaining text preserves leading zeros such as '007'. The `predict_fn`
    argument should be the existing `src.inference.predict_digit` function.
    """
    segments = segment_digits(
        image, max_digits=max_digits, expected_digits=expected_digits
    )
    predictions = []

    for segment in segments:
        if canvas_mode:
            processed, model_input = preprocess_canvas_segment(segment.image)
        else:
            processed, model_input = preprocess_image(segment.image)
        digit, scores = predict_fn(model_input)
        scores = np.asarray(scores, dtype=np.float32)
        if not 0 <= int(digit) <= 9 or scores.shape != (10,):
            raise ValueError("Unexpected digit classifier output.")
        predictions.append(
            {
                "digit": int(digit),
                "score": float(scores[int(digit)]),
                "crop": segment.image,
                "processed": processed,
                "box": segment.box,
            }
        )

    return "".join(str(item["digit"]) for item in predictions), predictions


def preprocess_canvas_segment(
    image: Image.Image,
) -> tuple[Image.Image, np.ndarray]:
    """Prepare one segmented canvas digit using the single-canvas method."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)

    # Convert black strokes on white into bright strokes on black.
    ink = np.clip(255.0 - gray, 0, 255).astype(np.uint8)
    ink[ink < 35] = 0

    digit = Image.fromarray(ink)
    bbox = digit.getbbox()

    if bbox is None:
        raise ValueError("No visible digit detected in the canvas region.")

    # Preserve every visible stroke, including disconnected parts.
    digit = digit.crop(bbox)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    processed = Image.new("L", (28, 28), 0)
    processed.paste(
        digit,
        (
            (28 - digit.width) // 2,
            (28 - digit.height) // 2,
        ),
    )

    model_input = np.asarray(processed, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

    return processed, model_input
