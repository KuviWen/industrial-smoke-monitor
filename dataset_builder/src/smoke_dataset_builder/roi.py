"""ROI parsing and image-cropping helpers shared by the dataset tools."""

from __future__ import annotations

from collections.abc import Sequence

Roi = tuple[int, int, int, int]


def parse_roi(value: str | Sequence[int] | None) -> Roi | None:
    """Parse ``x1,y1,x2,y2`` into an integer ROI.

    Coordinates use the original image's top-left origin.  ``x2`` and ``y2``
    are exclusive crop boundaries, just like a NumPy slice.  An empty string
    or ``None`` means that ROI cropping is disabled.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",")]
    else:
        parts = list(value)
    if len(parts) != 4:
        raise ValueError("ROI must use x1,y1,x2,y2, for example 600,0,1200,500")
    try:
        coordinates = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI coordinates must be integers: x1,y1,x2,y2") from exc
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "ROI must have positive width and height; use x1,y1,x2,y2, "
            "not x,y,width,height"
        )
    return coordinates  # type: ignore[return-value]


def validate_roi(roi: Roi, width: int, height: int) -> Roi:
    """Validate an ROI against one image's dimensions and return it."""

    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1, y1, x2, y2 = roi
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"ROI {format_roi(roi)} is outside image bounds 0,0,{width},{height}"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI must have positive width and height: {format_roi(roi)}")
    return roi


def format_roi(roi: Roi | None) -> str:
    """Return an ROI in the command-line/configuration format."""

    return "" if roi is None else ",".join(str(value) for value in roi)


def crop_frame(frame, roi: Roi):
    """Return a copy of ``frame`` cropped to a validated ROI."""

    if frame is None or getattr(frame, "ndim", 0) < 2:
        raise ValueError("frame must be a two-dimensional or three-dimensional array")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    x1, y1, x2, y2 = validate_roi(roi, width, height)
    cropped = frame[y1:y2, x1:x2]
    if getattr(cropped, "size", 0) == 0:
        raise ValueError(f"ROI produced an empty image: {format_roi(roi)}")
    return cropped.copy()
