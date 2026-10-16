"""Draw exactly one set of model detection boxes in source-image coordinates."""

from PIL import Image, ImageDraw


def annotate_regions(image: Image.Image, details: list[dict]) -> Image.Image:
    """Return an annotated RGB image at its original dimensions.

    The caller must pass the same orientation-corrected input supplied to the
    recognizer. Streamlit can scale this output for display afterward.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    stroke = max(2, round(min(width, height) / 250))

    for index, detail in enumerate(details, start=1):
        left, top, right, bottom = map(int, detail["box"])
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Detection box is outside the source image.")
        draw.rectangle((left, top, right - 1, bottom - 1), outline="red", width=stroke)
        draw.text((left, max(0, top - 18)), str(index), fill="red")

    return annotated
