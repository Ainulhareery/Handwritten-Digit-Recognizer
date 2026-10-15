"""Segment 2–5 clearly separated or lightly touching handwritten digits.

Experimental segmentation, not general document OCR. An optional `expected_digits`
lets a user supply the number of characters (not their identities).
"""

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DigitSegment:
    image: Image.Image
    box: tuple[int, int, int, int]


def _prepare(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")
    if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
    gray = image.convert("L")
    ow, oh = gray.size
    gray.thumbnail((1200, 800), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.uint8), (ow, oh)


def _segment_blue_ink(image, max_digits, expected_digits):
    """Use strongly blue pen strokes when they can be isolated confidently.

    A color difference helps separate blue handwriting from pale blue ruling.
    If the evidence is ambiguous, fall back to grayscale segmentation.
    """
    if image.mode not in ("RGB", "RGBA", "P"):
        return None

    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    rgb.thumbnail((1200, 800), Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.int16)
    height, width = pixels.shape[:2]

    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    # Reject pale paper ruling while retaining much darker blue pen strokes.
    blue_excess = blue - red
    mask = np.uint8((blue_excess >= 22) & (blue - green >= 12)) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(28, round(width * height * 0.00008))
    candidates = []
    for index in range(1, count):
        x, y, bw, bh, area = (int(value) for value in stats[index])
        if area < min_area or bh < max(10, height * 0.025):
            continue
        if bw < 4 or bw > width * 0.32 or bh > height * 0.46:
            continue
        candidates.append((x, y, bw, bh, area))

    if not candidates:
        return None

    # Only take a coherent line of similarly sized handwriting components.
    candidates.sort(key=lambda candidate: candidate[4], reverse=True)
    seed = candidates[0]
    seed_center = seed[1] + seed[3] / 2
    grouped = [
        part
        for part in candidates
        if abs(part[1] + part[3] / 2 - seed_center) <= max(16, 0.65 * seed[3])
        and part[3] >= 0.42 * seed[3]
        and part[3] <= 2.2 * seed[3]
        and part[4] >= 0.12 * seed[4]
    ]
    grouped.sort(key=lambda part: part[0])

    if not 2 <= len(grouped) <= max_digits:
        return None
    if expected_digits is not None and len(grouped) != expected_digits:
        return None

    # A large gap usually means unrelated writing or spurious image regions.
    heights = [part[3] for part in grouped]
    typical_height = float(np.median(heights))
    for first, second in zip(grouped[:-1], grouped[1:]):
        if second[0] - (first[0] + first[2]) > typical_height * 1.5:
            return None

    sx, sy = original_width / width, original_height / height
    result = []
    for x, y, bw, bh, _ in grouped:
        area = mask[y : y + bh, x : x + bw]
        pad = max(5, round(0.23 * max(bw, bh)))
        crop = np.full((bh + 2 * pad, bw + 2 * pad), 255, dtype=np.uint8)
        crop[pad : pad + bh, pad : pad + bw] = 255 - area
        result.append(
            DigitSegment(
                Image.fromarray(crop),
                (
                    round(x * sx),
                    round(y * sy),
                    round((x + bw) * sx),
                    round((y + bh) * sy),
                ),
            )
        )
    return result


def segment_digits(image, *, max_digits=5, expected_digits=None):
    if not 2 <= max_digits <= 20:
        raise ValueError("max_digits must be between 2 and 20.")
    if expected_digits is not None and not 2 <= expected_digits <= max_digits:
        raise ValueError("expected_digits must be between 2 and max_digits.")
    color_segments = _segment_blue_ink(image, max_digits, expected_digits)
    if color_segments is not None:
        return color_segments
    gray, (ow, oh) = _prepare(image)
    h, w = gray.shape
    if h < 24 or w < 24:
        raise ValueError("Image is too small for recognition.")
    # Local background removal works better than global Otsu on uneven paper.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(7, min(h, w) / 35))
    if np.median(gray) >= 128:
        contrast = cv2.subtract(background, gray)
        background_fill = 255
    else:
        contrast = cv2.subtract(gray, background)
        background_fill = 0
    # Strong handwritten strokes are dark relative to nearby paper.
    # Reject weak rule lines and most paper texture.
    positive = contrast[contrast > 5]
    if positive.size < 20:
        raise ValueError("No clear handwriting detected.")
    threshold = max(15, min(38, float(np.percentile(positive, 72)) * 0.55))
    mask = np.uint8(contrast > threshold) * 255
    # Remove thin rules spanning much of the image, not individual digit bars.
    long_width = max(35, round(w * 0.38))
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (long_width, 1))
    )
    mask = cv2.subtract(mask, horizontal)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(18, int(w * h * 0.00024))
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        if area < min_area or bh < max(9, h * 0.045) or bw < max(3, w * 0.008):
            continue
        if bw > 0.78 * w or bh > 0.75 * h:
            continue
        if bw > bh * 6:
            continue
        if (x == 0 or y == 0 or x + bw >= w or y + bh >= h) and area < 3 * min_area:
            continue
        candidates.append((i, x, y, bw, bh, area))
    if not candidates:
        raise ValueError("No usable handwriting regions detected.")
    # Select dominant writing rather than paper wrinkles and tiny marks.
    # Group components by the writing line's vertical center.
    interior = [
        c
        for c in candidates
        if c[1] > 0.015 * w
        and c[1] + c[3] < 0.985 * w
        and c[2] > 0.015 * h
        and c[2] + c[4] < 0.985 * h
    ]
    eligible = interior or candidates
    peak_area = max(c[5] for c in eligible)
    dominant = [c for c in eligible if c[5] >= 0.23 * peak_area]
    best_group = []
    best_mass = -1
    for seed in dominant:
        _, _, sy, _, sh, _ = seed
        cy = sy + sh * 0.5
        group = [
            c for c in dominant if abs(c[2] + c[4] * 0.5 - cy) <= max(24, 0.12 * h)
        ]
        mass = sum(c[5] for c in group)
        if mass > best_mass:
            best_group, best_mass = group, mass
    # Allow nearby detached pen strokes with same text line.
    left = max(0, min(c[1] for c in best_group) - round(w * 0.006))
    right = min(w, max(c[1] + c[3] for c in best_group) + round(w * 0.006))
    top = max(0, min(c[2] for c in best_group) - round(h * 0.012))
    bottom = min(h, max(c[2] + c[4] for c in best_group) + round(h * 0.012))
    if right - left < 20 or bottom - top < 15:
        raise ValueError("Unable to isolate a handwritten number.")
    # Keep components whose centers align with dominant writing area, but do not
    # demand that the whole number is connected (important for disconnected strokes).
    roi = mask[top:bottom, left:right]
    raw = np.sum(roi > 0, axis=0).astype(np.float32)
    if np.count_nonzero(raw) < 12:
        raise ValueError("Insufficient digit strokes to segment.")
    first, last = np.flatnonzero(raw > 0)[[0, -1]]
    raw = raw[first : last + 1]
    left += int(first)
    right = left + len(raw)
    width = len(raw)
    if width < 20:
        raise ValueError("The detected number is too narrow.")
    sigma = max(1.2, width / 120)
    smooth = cv2.GaussianBlur(raw.reshape(1, -1), (0, 0), sigmaX=sigma)[0]
    # Choose vertical cuts at low-ink valleys; cuts may cross a small overlap.
    minimum_spacing = max(7, round(width / (max_digits * 1.65)))
    local_min = []
    for j in range(minimum_spacing, width - minimum_spacing):
        if smooth[j] <= smooth[j - 1] and smooth[j] <= smooth[j + 1]:
            local_min.append(j)
    # Remove near-duplicate minima (flat valleys) via greedy scoring.
    density = float(np.percentile(smooth, 65))
    if expected_digits is None:
        # One split per low-ink *interval*, not one per pixel-wise minimum.
        # This avoids splitting a wide blank gap into several fake digits.
        low = smooth < max(2.0, density * 0.50)
        indexes = np.flatnonzero(low)
        groups = (
            np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
            if indexes.size
            else []
        )
        cuts = []
        for group in groups:
            if group.size == 0:
                continue
            center = int(group[len(group) // 2])
            if minimum_spacing <= center <= width - minimum_spacing:
                cuts.append(center)
        # Too many plausible boundaries means we cannot identify digits safely.
        if not 1 <= len(cuts) <= max_digits - 1:
            raise ValueError(
                f"Found {len(cuts) + 1} candidate digit regions; expected 2–{max_digits}. "
                "Try a tighter crop or select the expected digit count."
            )
    else:
        # With number of digits known, search all candidate cut combinations with
        # a mild width regularizer to prevent tiny fragments.
        from itertools import combinations

        count = expected_digits
        pool = sorted(local_min, key=lambda j: float(smooth[j]))[:35]
        if len(pool) < count - 1:
            raise ValueError("Not enough potential digit boundaries.")
        desired = width / count
        best = None
        cuts = None
        for proposed in combinations(sorted(pool), count - 1):
            spans = np.diff([0, *proposed, width])
            if np.min(spans) < max(6, width / (count * 2.5)):
                continue
            score = sum(float(smooth[j]) for j in proposed) + density * 0.40 * sum(
                ((span - desired) / desired) ** 2 for span in spans
            )
            if best is None or score < best:
                best = score
                cuts = list(proposed)
        if cuts is None:
            raise ValueError(
                "Cannot identify plausible splits. Write the digits with more space."
            )
    edges = [0, *cuts, width]
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < max(5, width / (max_digits * 3)):
            raise ValueError("Detected an implausibly narrow digit region.")
        part = mask[top:bottom, left + a : left + b]
        ys, xs = np.where(part > 0)
        if ys.size < 10:
            raise ValueError("A proposed digit region contains too little ink.")
        x0 = left + a + int(xs.min())
        x1 = left + a + int(xs.max()) + 1
        y0 = top + int(ys.min())
        y1 = top + int(ys.max()) + 1
        # Binarized crop eliminates ruled-paper/shadow artifacts before the
        # per-digit pipeline; use visible foreground for white-paper style.
        data = part[
            int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1
        ]
        # NB: x is local to the segment.
        pad = max(5, round(0.23 * max(data.shape)))
        crop = np.full(
            (data.shape[0] + 2 * pad, data.shape[1] + 2 * pad),
            background_fill,
            dtype=np.uint8,
        )
        if background_fill == 255:
            crop[pad : pad + data.shape[0], pad : pad + data.shape[1]] = 255 - data
        else:
            crop[pad : pad + data.shape[0], pad : pad + data.shape[1]] = data
        sx = ow / w
        sy = oh / h
        result.append(
            DigitSegment(
                Image.fromarray(crop),
                (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)),
            )
        )
    if not 2 <= len(result) <= max_digits:
        raise ValueError(f"Detected {len(result)} regions, expected 2–{max_digits}.")
    return result
