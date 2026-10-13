
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf


# Locate the trained model from the project root
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "final_digit_model.keras"
)

EXPECTED_INPUT_SHAPE = (1, 28, 28, 1)
NUMBER_OF_CLASSES = 10


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained CNN once and reuse it.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if tuple(model.input_shape[1:]) != (28, 28, 1):
        raise ValueError(
            "The model input shape is incompatible "
            "with 28x28 grayscale images."
        )

    if model.output_shape[-1] != NUMBER_OF_CLASSES:
        raise ValueError(
            "The model must output scores for 10 digits."
        )

    return model


def predict_digit(model_input):
    """
    Predict a single digit from a preprocessed image.

    Args:
        model_input: NumPy array shaped (1, 28, 28, 1),
                     normalized to the range [0, 1].

    Returns:
        predicted_digit: Integer from 0 to 9.
        scores: Model output scores for digits 0 through 9.
    """

    array = np.asarray(model_input, dtype=np.float32)

    # Validate the input before passing it to the model
    if array.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Expected input shape {EXPECTED_INPUT_SHAPE}, "
            f"but received {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "The input contains invalid numeric values."
        )

    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(
            "Pixel values must be normalized between 0 and 1."
        )

    model = load_model()

    outputs = model.predict(array, verbose=0)

    if outputs.shape != (1, NUMBER_OF_CLASSES):
        raise ValueError(
            "Unexpected model output shape."
        )

    scores = outputs[0]
    predicted_digit = int(np.argmax(scores))

    return predicted_digit, scores
