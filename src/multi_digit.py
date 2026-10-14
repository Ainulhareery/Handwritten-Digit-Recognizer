"""Classify each segmented digit using the project's existing CNN."""

from typing import Callable

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image
from src.segmentation import segment_digits


def recognize_number(
    image: Image.Image,
    *,
    predict_fn: Callable,
    max_digits: int = 5,
) -> tuple[str, list[dict]]:
    """Return the full number as *text* and diagnostic data for each digit.

    Retaining text preserves leading zeros such as '007'. The `predict_fn`
    argument should be the existing `src.inference.predict_digit` function.
    """
    segments = segment_digits(image, max_digits=max_digits)
    predictions = []

    for segment in segments:
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
