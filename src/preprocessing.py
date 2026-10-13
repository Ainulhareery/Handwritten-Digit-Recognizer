
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Prepare a handwritten digit as a 28x28 MNIST-style image."""

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Convert to grayscale and enhance contrast
    gray = ImageOps.grayscale(image)
    low, high = gray.getextrema()

    if high - low < 12:
        raise ValueError("No visible digit detected. Upload a clearer image.")

    gray = ImageOps.autocontrast(gray)

    # MNIST expects bright digits on a dark background
    if np.asarray(gray).mean() > 127:
        gray = ImageOps.invert(gray)

    pixels = np.asarray(gray)

    # Separate dark handwriting from the paper background
    mask = pixels > 120
    height, width = mask.shape

    visited = set()
    best = []

    # Find connected strokes and identify the main digit
    for row, col in zip(*np.nonzero(mask)):
        start = (int(row), int(col))

        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    neighbor = (ny, nx)

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        stack.append(neighbor)

        rows, cols = zip(*component)
        component_width = max(cols) - min(cols) + 1
        component_height = max(rows) - min(rows) + 1

        # Ignore long horizontal notebook lines
        if (
            component_width > width * 0.5
            and component_height < height * 0.08
        ):
            continue

        if len(component) > len(best):
            best = component

    if not best:
        raise ValueError("No visible digit detected. Upload a clearer image.")

    # Keep the digit and remove surrounding distractions
    rows, cols = np.array(best).T
    isolated = np.zeros_like(pixels)
    isolated[rows, cols] = pixels[rows, cols]

    bbox = (
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1,
    )

    digit = Image.fromarray(isolated).crop(bbox)

    # Scale digit to fit inside a 20x20 area
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center on an MNIST-sized 28x28 canvas
    processed = Image.new("L", (28, 28), 0)
    processed.paste(
        digit,
        (
            (28 - digit.width) // 2,
            (28 - digit.height) // 2,
        ),
    )

    # Normalize pixels for the CNN
    model_input = np.asarray(processed, dtype=np.float32)
    model_input = (model_input / 255.0).reshape(1, 28, 28, 1)

    return processed, model_input
