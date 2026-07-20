"""YOLO11 instance-segmentation label helpers.

The parent project uses one class, ``smoke`` (class id 0).  A segmentation
label is one line per polygon::

    0 x1 y1 x2 y2 ...

All coordinates after the class id are normalized to the interval [0, 1].
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def polygon_to_yolo_line(
    polygon: Polygon,
    width: int,
    height: int,
    class_id: int = 0,
) -> str | None:
    """Convert image-pixel polygon points to one YOLO segmentation line.

    Invalid polygons (fewer than three usable points or zero area) return
    ``None`` so the caller can safely ignore an accidental click sequence.
    Points are clipped to the image rectangle before normalization.
    """

    if width <= 0 or height <= 0 or class_id < 0:
        raise ValueError("width, height, and class_id must be positive")

    points: list[Point] = []
    for x, y in polygon:
        x_float = float(x)
        y_float = float(y)
        if not (math.isfinite(x_float) and math.isfinite(y_float)):
            continue
        clipped = (
            min(max(x_float, 0.0), float(width - 1)),
            min(max(y_float, 0.0), float(height - 1)),
        )
        if not points or clipped != points[-1]:
            points.append(clipped)

    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or _polygon_area(points) <= 1e-6:
        return None

    coordinates = " ".join(
        f"{x / width:.6f} {y / height:.6f}" for x, y in points
    )
    return f"{class_id} {coordinates}"


def polygons_to_yolo_lines(
    polygons: Iterable[Polygon],
    width: int,
    height: int,
    class_id: int = 0,
) -> list[str]:
    """Convert all valid polygons to YOLO lines."""

    lines: list[str] = []
    for polygon in polygons:
        line = polygon_to_yolo_line(polygon, width, height, class_id)
        if line is not None:
            lines.append(line)
    return lines


def flip_polygons_horizontally(
    polygons: Iterable[Polygon],
    width: int,
) -> list[list[Point]]:
    """Mirror image-pixel polygons around the vertical center line.

    Polygon points in the annotation GUI are expressed in image pixels.  For
    an image whose valid x coordinates are ``0 .. width-1``, a horizontal
    flip maps ``x`` to ``width - 1 - x`` while leaving y unchanged.
    """

    if width <= 0:
        raise ValueError("width must be positive")
    return [
        [(float(width - 1) - float(x), float(y)) for x, y in polygon]
        for polygon in polygons
    ]


def validate_yolo_line(line: str, class_count: int = 1) -> str | None:
    """Return a human-readable error, or ``None`` when a line is valid."""

    tokens = line.split()
    if not tokens:
        return "empty line"
    try:
        class_id = int(tokens[0])
    except ValueError:
        return "class id is not an integer"
    if not 0 <= class_id < class_count:
        return f"class id {class_id} is outside 0..{class_count - 1}"
    if len(tokens[1:]) < 6 or len(tokens[1:]) % 2:
        return "a polygon needs at least three x/y pairs"
    try:
        coordinates = [float(value) for value in tokens[1:]]
    except ValueError:
        return "a coordinate is not a number"
    if not all(math.isfinite(value) for value in coordinates):
        return "a coordinate is not finite"
    if not all(-1e-6 <= value <= 1.000001 for value in coordinates):
        return "a normalized coordinate is outside [0, 1]"
    return None


def validate_yolo_lines(lines: Iterable[str], class_count: int = 1) -> list[str]:
    """Validate non-empty label lines and return indexed error messages."""

    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        error = validate_yolo_line(stripped, class_count)
        if error:
            errors.append(f"line {line_number}: {error}")
    return errors


def mask_to_polygons(
    mask,
    min_area: float = 20.0,
    epsilon_ratio: float = 0.002,
) -> list[list[Point]]:
    """Extract simplified pixel polygons from a binary mask.

    OpenCV is imported lazily so label-only utilities remain usable in a
    documentation or CI environment where the optional native module is not
    installed yet.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("mask_to_polygons requires opencv-python and numpy") from exc

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    binary = (array > 0).astype("uint8") * 255
    contours_result = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    polygons: list[list[Point]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.5, perimeter * epsilon_ratio)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        points = [
            (float(point[0][0]), float(point[0][1])) for point in simplified
        ]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def result_to_polygons(
    result,
    image_shape: tuple[int, ...],
    class_id: int = 0,
    min_area: float = 20.0,
) -> list[list[Point]]:
    """Read class-filtered polygons from one Ultralytics segmentation result."""

    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    boxes = getattr(result, "boxes", None)
    classes = None
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()

    polygons: list[list[Point]] = []
    xy_polygons = getattr(masks, "xy", None)
    if xy_polygons is not None:
        for index, polygon_array in enumerate(xy_polygons):
            if classes is not None and index < len(classes) and classes[index] != class_id:
                continue
            points = [
                (float(point[0]), float(point[1]))
                for point in polygon_array
                if len(point) >= 2
            ]
            if len(points) >= 3 and _polygon_area(points) >= min_area:
                polygons.append(points)
        if polygons:
            return polygons

    # Fallback for result objects that expose only masks.data.
    data = getattr(masks, "data", None)
    if data is None:
        return polygons
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("result_to_polygons requires opencv-python and numpy") from exc

    height, width = int(image_shape[0]), int(image_shape[1])
    for index, mask in enumerate(data.detach().cpu().numpy()):
        if classes is not None and index < len(classes) and classes[index] != class_id:
            continue
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        polygons.extend(mask_to_polygons(resized, min_area=min_area))
    return polygons
