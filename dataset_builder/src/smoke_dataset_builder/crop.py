"""Crop a YOLO segmentation dataset to an ROI without changing its source."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS, SPLITS, write_dataset_yaml
from .roi import Roi, format_roi, validate_roi
from .yolo import polygon_to_yolo_line


@dataclass(frozen=True)
class CropSummary:
    """Counts returned by :func:`crop_dataset`."""

    images: int
    polygons: int
    negatives: int
    output: Path


def _require_cv2_numpy():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("ROI dataset cropping requires opencv-python and numpy") from exc
    return cv2, np


def _read_polygon_lines(label_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Read YOLO segmentation lines as class ids and normalized points."""

    polygons: list[tuple[int, list[tuple[float, float]]]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 7 or len(tokens[1:]) % 2:
            raise ValueError(
                f"Invalid segmentation label {label_path} line {line_number}: "
                "a polygon needs at least three x/y pairs"
            )
        try:
            class_id = int(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Invalid segmentation label {label_path} line {line_number}: "
                "class id and coordinates must be numbers"
            ) from exc
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"Invalid segmentation label {label_path} line {line_number}: "
                "normalized coordinates must be in [0, 1]"
            )
        points = list(zip(coordinates[::2], coordinates[1::2]))
        polygons.append((class_id, points))
    return polygons


def _cropped_polygon_lines(
    label_path: Path,
    *,
    image_width: int,
    image_height: int,
    roi: Roi,
    min_mask_area: float,
) -> list[str]:
    """Clip every polygon by rasterizing its mask inside the ROI.

    Geometric point filtering is not sufficient when an ROI cuts through an
    edge: a polygon vertex can be outside the ROI while the polygon itself
    still intersects it.  Rasterizing each instance mask and extracting its
    cropped contours handles both edge intersections and masks that become
    multiple components after cropping.
    """

    cv2, np = _require_cv2_numpy()
    x1, y1, x2, y2 = validate_roi(roi, image_width, image_height)
    cropped_width, cropped_height = x2 - x1, y2 - y1
    lines: list[str] = []
    for class_id, normalized_points in _read_polygon_lines(label_path):
        points = np.asarray(
            [
                [round(x * image_width) - x1, round(y * image_height) - y1]
                for x, y in normalized_points
            ],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue
        mask = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        contours_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        for contour in contours:
            if float(cv2.contourArea(contour)) < min_mask_area:
                continue
            contour_points = [
                (float(point[0][0]), float(point[0][1])) for point in contour
            ]
            line = polygon_to_yolo_line(
                contour_points,
                cropped_width,
                cropped_height,
                class_id=class_id,
            )
            if line is not None:
                lines.append(line)
    return lines


def _read_manifest(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row.get("split", ""), Path(row.get("image", "")).stem): row
            for row in csv.DictReader(handle)
        }


def _output_is_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _write_manifest(
    output: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    fields = (
        "image",
        "label",
        "split",
        "source_image",
        "source_video",
        "frame_index",
        "timestamp_seconds",
        "label_count",
        "label_status",
        "roi_xyxy",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def crop_dataset(
    input_root: str | Path,
    output_root: str | Path,
    roi: Roi,
    *,
    overwrite: bool = False,
    min_mask_area: float = 1.0,
) -> CropSummary:
    """Create a separate ROI-cropped YOLO dataset.

    The input directory is read only.  Images are cropped and each YOLO
    segmentation polygon is converted through a raster mask before being
    re-normalized to the cropped image dimensions.  The output must not be
    the input directory itself.
    """

    if min_mask_area < 0:
        raise ValueError("min_mask_area must be non-negative")
    cv2, _np = _require_cv2_numpy()
    source = Path(input_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input dataset not found: {source}")
    if source == output:
        raise ValueError("Output dataset must be different from the input dataset")
    if _output_is_nonempty(output) and not overwrite:
        raise FileExistsError(
            f"Output is not empty; choose a new directory or use --overwrite only "
            f"for this exact output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(source)
    manifest_rows: list[dict[str, object]] = []
    image_count = 0
    polygon_count = 0
    negative_count = 0

    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        target_image_dir = output / "images" / split
        target_label_dir = output / "labels" / split
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"OpenCV could not read image: {image_path}")
            image_height, image_width = image.shape[:2]
            validate_roi(roi, int(image_width), int(image_height))
            x1, y1, x2, y2 = roi
            cropped_image = image[y1:y2, x1:x2].copy()
            if cropped_image.size == 0:
                raise ValueError(f"ROI produced an empty image for {image_path}")
            lines = _cropped_polygon_lines(
                label_path,
                image_width=int(image_width),
                image_height=int(image_height),
                roi=roi,
                min_mask_area=min_mask_area,
            )
            target_image = target_image_dir / image_path.name
            target_label = target_label_dir / f"{image_path.stem}.txt"
            if (target_image.exists() or target_label.exists()) and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {target_image} or {target_label}"
                )
            if not cv2.imwrite(str(target_image), cropped_image):
                raise IOError(f"OpenCV could not write image: {target_image}")
            target_label.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            original = manifest.get((split, image_path.stem), {})
            source_image = str(image_path.relative_to(source)).replace("\\", "/")
            output_image = str(target_image.relative_to(output)).replace("\\", "/")
            output_label = str(target_label.relative_to(output)).replace("\\", "/")
            old_status = original.get("label_status", "")
            manifest_rows.append(
                {
                    "image": output_image,
                    "label": output_label,
                    "split": split,
                    "source_image": source_image,
                    "source_video": original.get("source_video", ""),
                    "frame_index": original.get("frame_index", ""),
                    "timestamp_seconds": original.get("timestamp_seconds", ""),
                    "label_count": len(lines),
                    "label_status": f"{old_status};roi_cropped" if old_status else "roi_cropped",
                    "roi_xyxy": format_roi(roi),
                }
            )
            image_count += 1
            polygon_count += len(lines)
            if not lines:
                negative_count += 1

    if image_count == 0:
        raise FileNotFoundError(
            f"No supported images found under {source / 'images'}"
        )
    write_dataset_yaml(output)
    _write_manifest(output, manifest_rows)
    return CropSummary(image_count, polygon_count, negative_count, output)
