"""Write and inspect the YOLO directory layout consumed by the parent app."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .yolo import Polygon, flip_polygons_horizontally, polygons_to_yolo_lines

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("dataset writing requires opencv-python") from exc
    return cv2


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "sample"


def _unique_stem(images_dir: Path, labels_dir: Path, requested: str) -> str:
    stem = safe_stem(requested)
    if not any((images_dir / f"{stem}{extension}").exists() for extension in IMAGE_EXTENSIONS) and not (
        labels_dir / f"{stem}.txt"
    ).exists():
        return stem
    counter = 1
    while True:
        candidate = f"{stem}_{counter:03d}"
        if not any(
            (images_dir / f"{candidate}{extension}").exists()
            for extension in IMAGE_EXTENSIONS
        ) and not (labels_dir / f"{candidate}.txt").exists():
            return candidate
        counter += 1


def dataset_yaml_content(root: str | Path, class_name: str = "smoke") -> dict:
    """Return a YAML-compatible dataset configuration."""

    root_path = Path(root).expanduser().resolve()
    return {
        "path": root_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }


def write_dataset_yaml(root: str | Path, class_name: str = "smoke") -> Path:
    """Create or refresh ``dataset.yaml`` and return its path."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("writing dataset.yaml requires PyYAML") from exc

    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (root_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (root_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    yaml_path = root_path / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            dataset_yaml_content(root_path, class_name),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return yaml_path


def append_manifest_record(root: str | Path, record: Mapping[str, object]) -> None:
    """Append one provenance record to the dataset's CSV manifest."""

    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / "manifest.csv"
    default_fields = (
        "image",
        "label",
        "split",
        "source_video",
        "frame_index",
        "timestamp_seconds",
        "label_count",
        "label_status",
        "roi_xyxy",
    )
    is_new = not manifest_path.exists() or manifest_path.stat().st_size == 0
    if is_new:
        with manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=default_fields)
            writer.writeheader()
            writer.writerow({field: record.get(field, "") for field in default_fields})
        return

    # Older datasets did not have ``roi_xyxy``.  Upgrade only the manifest
    # header/rows when needed so adding ROI support remains backward compatible.
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fields = list(reader.fieldnames or default_fields)
        rows = list(reader)
    fields = existing_fields + [field for field in default_fields if field not in existing_fields]
    if fields != existing_fields:
        temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(manifest_path)
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerow({field: record.get(field, "") for field in fields})


class YoloDatasetWriter:
    """Persist reviewed frames as YOLO11 segmentation samples."""

    def __init__(self, root: str | Path, class_name: str = "smoke") -> None:
        self.root = Path(root).expanduser().resolve()
        self.class_name = class_name
        write_dataset_yaml(self.root, class_name)

    def save_sample(
        self,
        image,
        polygons: Iterable[Polygon],
        split: str,
        stem: str,
        *,
        source_video: str | Path = "",
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        generate_horizontal_flip: bool = False,
        roi_xyxy: str = "",
    ) -> tuple[Path, Path]:
        """Save a BGR image and its reviewed polygon label.

        An empty ``polygons`` iterable deliberately creates an empty label
        file, which is the YOLO representation of a confirmed negative frame.

        The returned paths always refer to the original sample.  When
        ``generate_horizontal_flip`` is true, a second image/label pair with
        the ``_flip`` suffix is written to the same split and manifest.
        """

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if image is None or getattr(image, "ndim", 0) < 2:
            raise ValueError("image must be a two-dimensional or three-dimensional array")

        cv2 = _require_cv2()
        height, width = int(image.shape[0]), int(image.shape[1])
        polygon_list = [list(polygon) for polygon in polygons]
        image_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        actual_stem = _unique_stem(image_dir, label_dir, stem)
        image_path, label_path, _ = self._write_one(
            cv2=cv2,
            image=image,
            polygons=polygon_list,
            width=width,
            height=height,
            split=split,
            requested_stem=actual_stem,
            source_video=source_video,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            label_status="reviewed",
            roi_xyxy=roi_xyxy,
        )

        if generate_horizontal_flip:
            flipped_image = cv2.flip(image, 1)
            flipped_polygons = flip_polygons_horizontally(polygon_list, width)
            self._write_one(
                cv2=cv2,
                image=flipped_image,
                polygons=flipped_polygons,
                width=width,
                height=height,
                split=split,
                requested_stem=f"{actual_stem}_flip",
                source_video=source_video,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                label_status="reviewed_augmented_horizontal_flip",
                roi_xyxy=roi_xyxy,
            )
        return image_path, label_path

    def _write_one(
        self,
        *,
        cv2,
        image,
        polygons: Iterable[Polygon],
        width: int,
        height: int,
        split: str,
        requested_stem: str,
        source_video: str | Path,
        frame_index: int | None,
        timestamp_seconds: float | None,
        label_status: str,
        roi_xyxy: str,
    ) -> tuple[Path, Path, int]:
        """Write one image/label pair and one provenance row."""

        image_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        actual_stem = _unique_stem(image_dir, label_dir, requested_stem)
        image_path = image_dir / f"{actual_stem}.jpg"
        label_path = label_dir / f"{actual_stem}.txt"
        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(f"OpenCV could not write image: {image_path}")

        lines = polygons_to_yolo_lines(polygons, width, height, class_id=0)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        append_manifest_record(
            self.root,
            {
                "image": str(image_path.relative_to(self.root)).replace("\\", "/"),
                "label": str(label_path.relative_to(self.root)).replace("\\", "/"),
                "split": split,
                "source_video": str(source_video),
                "frame_index": "" if frame_index is None else frame_index,
                "timestamp_seconds": ""
                if timestamp_seconds is None
                else f"{timestamp_seconds:.3f}",
                "label_count": len(lines),
                "label_status": label_status,
                "roi_xyxy": roi_xyxy,
            },
        )
        return image_path, label_path, len(lines)
