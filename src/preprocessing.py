import numpy as np
from PIL import Image, ImageFilter


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Isolate a handwritten digit and convert it to MNIST-style input."""
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    gray = image.convert("L")
    gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    height, width = pixels.shape

    # Detect ink polarity before changing the image contrast.
    median = np.median(pixels)
    dark_tail = median - np.percentile(pixels, 1)
    light_tail = np.percentile(pixels, 99) - median
    dark_ink = dark_tail >= light_tail

    # A blurred copy approximates the paper/background lighting.
    blurred = gray.filter(
        ImageFilter.GaussianBlur(radius=max(5, min(height, width) / 12))
    )
    background = np.asarray(blurred, dtype=np.float32)
    contrast = background - pixels if dark_ink else pixels - background
    threshold = max(8.0, 0.35 * float(np.percentile(contrast, 99.7)))
    mask = contrast > threshold

    # Identify connected strokes. Ignore notebook lines and edge artifacts.
    visited = np.zeros((height, width), dtype=bool)
    best_component = None
    best_score = -1.0

    for row, col in zip(*np.nonzero(mask)):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        component = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ):
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        points = np.asarray(component)
        ys, xs = points[:, 0], points[:, 1]
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)

        if box_width > width * 0.55 and box_height < height * 0.10:
            continue
        if len(component) < max(12, int(height * width * 0.0005)):
            continue
        if ys.min() < height * 0.02 and box_height < height * 0.08:
            continue
        if ys.max() > height * 0.98 and box_height < height * 0.08:
            continue

        center_x = (xs.min() + xs.max()) / 2
        center_y = (ys.min() + ys.max()) / 2
        distance = np.hypot(
            (center_x - width / 2) / (width / 2),
            (center_y - height / 2) / (height / 2),
        )
        score = len(component) / (1 + 0.7 * distance)
        if score > best_score:
            best_score = score
            best_component = points

    if best_component is None:
        raise ValueError("No visible digit detected. Upload a clearer image.")

    ys, xs = best_component[:, 0], best_component[:, 1]
    digit_pixels = np.zeros((height, width), dtype=np.uint8)
    strengths = contrast[ys, xs]
    peak = max(30.0, float(np.percentile(strengths, 98)))
    values = np.clip(
        (strengths - threshold * 0.35) / (peak - threshold * 0.35),
        0, 1,
    ) * 255
    digit_pixels[ys, xs] = values.astype(np.uint8)

    digit = Image.fromarray(
        digit_pixels[
            int(ys.min()): int(ys.max()) + 1,
            int(xs.min()): int(xs.max()) + 1,
        ]
    )
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    digit = Image.blend(digit, digit.filter(ImageFilter.MaxFilter(3)), 0.45)

    processed = Image.new("L", (28, 28), 0)
    processed.paste(
        digit,
        ((28 - digit.width) // 2, (28 - digit.height) // 2),
    )
    model_input = np.asarray(processed, dtype=np.float32)
    model_input = model_input.reshape(1, 28, 28, 1) / 255.0
    return processed, model_input
