
import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 28
PIXEL_THRESHOLD = 40


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Convert an uploaded image into a normalized MNIST-shaped input.

    Returns:
        processed_image: 28x28 grayscale PIL image.
        model_input: float32 array with shape (1, 28, 28, 1).
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    # Convert typical dark-on-light handwriting to
    # bright strokes on a dark background.
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = image.point(
        lambda pixel: 255 if pixel > PIXEL_THRESHOLD else 0
    )

    bbox = image.getbbox()

    if bbox is None:
        raise ValueError(
            "No visible digit detected. Please upload a "
            "clear image containing one handwritten digit."
        )

    digit = image.crop(bbox)
    width, height = digit.size

    # Preserve the digit's aspect ratio and add padding.
    side = max(width, height) + 8
    canvas = Image.new("L", (side, side), 0)

    canvas.paste(
        digit,
        ((side - width) // 2, (side - height) // 2),
    )

    canvas.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS,
    )

    processed_image = Image.new("L", (28, 28), 0)
    processed_image.paste(
        canvas,
        (
            (28 - canvas.width) // 2,
            (28 - canvas.height) // 2,
        ),
    )

    pixels = np.asarray(
        processed_image,
        dtype=np.float32,
    ) / 255.0

    model_input = pixels[np.newaxis, ..., np.newaxis]

    return processed_image, model_input
