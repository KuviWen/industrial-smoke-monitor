"""Alert discovery, manual relabeling, and dataset export.

The reviewer deliberately depends only on OpenCV, NumPy, and PyYAML.  It does
not load YOLO weights.  This
keeps review safe on a field PC and makes it clear that a human, not a model,
decides the corrected label.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2

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
    """Convert a manually drawn pixel polygon to one YOLO-seg line."""

    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    points: list[Point] = []
    for x, y in polygon:
        x_value = float(x)
        y_value = float(y)
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        point = (
            min(max(x_value, 0.0), float(width - 1)),
            min(max(y_value, 0.0), float(height - 1)),
        )
        if not points or point != points[-1]:
            points.append(point)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or _polygon_area(points) <= 1e-6:
        return None
    coordinates = " ".join(
        f"{x / width:.6f} {y / height:.6f}" for x, y in points
    )
    return f"{class_id} {coordinates}"


def polygons_to_yolo_lines(
    polygons: Iterable[Polygon], width: int, height: int
) -> list[str]:
    return [
        line
        for polygon in polygons
        if (line := polygon_to_yolo_line(polygon, width, height)) is not None
    ]


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "alert"


def _unique_stem(images_dir: Path, labels_dir: Path, requested: str) -> str:
    stem = _safe_stem(requested)
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter:03d}"
        candidate = f"{stem}{suffix}"
        if not (images_dir / f"{candidate}.jpg").exists() and not (
            labels_dir / f"{candidate}.txt"
        ).exists():
            return candidate
        counter += 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class AlertItem:
    image_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def alert_id(self) -> str:
        return self.image_path.stem

    @property
    def review_status(self) -> str:
        return str(self.metadata.get("review", {}).get("status", "unreviewed"))


class AlertReviewService:
    """Manage alert sidecars and export corrected samples."""

    def __init__(self, alert_dir: str | Path) -> None:
        self.alert_dir = Path(alert_dir).expanduser().resolve()

    def list_alerts(self) -> list[AlertItem]:
        if not self.alert_dir.exists():
            return []
        items: list[AlertItem] = []
        for metadata_path in sorted(self.alert_dir.rglob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            image_path = self._image_path(metadata_path, metadata)
            if image_path is None or not image_path.is_file():
                continue
            items.append(AlertItem(image_path, metadata_path, metadata))
        return sorted(
            items,
            key=lambda item: str(item.metadata.get("timestamp_utc", item.image_path.name)),
        )

    @staticmethod
    def _image_path(metadata_path: Path, metadata: dict[str, Any]) -> Path | None:
        raw_path = metadata.get("image_path")
        if raw_path:
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = metadata_path.parent / candidate
            if candidate.is_file():
                return candidate.resolve()
        sibling = metadata_path.with_suffix(".jpg")
        return sibling.resolve() if sibling.is_file() else None

    @staticmethod
    def image_size(item: AlertItem) -> tuple[int, int]:
        image = cv2.imread(str(item.image_path))
        if image is None:
            raise IOError(f"Unable to read alert image: {item.image_path}")
        height, width = image.shape[:2]
        return width, height

    @staticmethod
    def initial_polygons(item: AlertItem) -> list[list[Point]]:
        review = item.metadata.get("review", {})
        if isinstance(review, dict) and isinstance(review.get("polygons"), list):
            return AlertReviewService._coerce_polygons(review["polygons"])
        polygons: list[list[Point]] = []
        instances = item.metadata.get("instances", [])
        if not isinstance(instances, list):
            return polygons
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            polygon = instance.get("polygon", [])
            points = AlertReviewService._coerce_polygons([polygon])
            polygons.extend(points)
        return polygons

    @staticmethod
    def _coerce_polygons(value: Any) -> list[list[Point]]:
        result: list[list[Point]] = []
        if not isinstance(value, list):
            return result
        for polygon in value:
            if not isinstance(polygon, (list, tuple)):
                continue
            points: list[Point] = []
            for point in polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        points.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue
            if len(points) >= 3:
                result.append(points)
        return result

    def keep_in_place(
        self,
        item: AlertItem,
        label: str,
        polygons: Iterable[Polygon],
    ) -> AlertItem:
        return self._write_review(item, label, polygons)

    def move_to_dataset(
        self,
        item: AlertItem,
        label: str,
        polygons: Iterable[Polygon],
        dataset_root: str | Path,
        split: str = "train",
    ) -> tuple[Path, Path]:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        root = Path(dataset_root).expanduser().resolve()
        try:
            root.relative_to(self.alert_dir)
        except ValueError:
            pass
        else:
            raise ValueError("training dataset must not be inside the alert directory")

        polygon_list = [list(polygon) for polygon in polygons]
        reviewed = self._write_review(item, label, polygon_list)
        image = cv2.imread(str(reviewed.image_path))
        if image is None:
            raise IOError(f"Unable to read alert image: {reviewed.image_path}")
        height, width = image.shape[:2]
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        self._write_dataset_yaml(root)

        stem = _unique_stem(image_dir, label_dir, f"alert_{reviewed.alert_id}")
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(f"Unable to write dataset image: {image_path}")
        review_polygons = [] if label == "no_smoke" else polygon_list
        label_lines = polygons_to_yolo_lines(review_polygons, width, height)
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        self._append_manifest(
            root,
            {
                "image": str(image_path.relative_to(root)).replace("\\", "/"),
                "label": str(label_path.relative_to(root)).replace("\\", "/"),
                "split": split,
                "source_video": reviewed.metadata.get(
                    "source_video", reviewed.metadata.get("source", "monitor_alert")
                ),
                "frame_index": reviewed.metadata.get("frame_index", ""),
                "timestamp_seconds": reviewed.metadata.get("timestamp_seconds", ""),
                "label_count": len(label_lines),
                "label_status": f"alert_reviewer_{label}",
            },
        )

        # The dataset now owns the reviewed copy.  Remove only the alert
        # artifact and its sidecar; unrelated runtime logs remain untouched.
        self._remove_alert_files(reviewed)
        return image_path, label_path

    def delete(self, item: AlertItem) -> None:
        self._remove_alert_files(item)

    def _write_review(
        self,
        item: AlertItem,
        label: str,
        polygons: Iterable[Polygon],
    ) -> AlertItem:
        if label not in {"smoke", "no_smoke"}:
            raise ValueError("label must be smoke or no_smoke")
        width, height = self.image_size(item)
        review_polygons = [] if label == "no_smoke" else [list(p) for p in polygons]
        updated = dict(item.metadata)
        updated["smoke"] = label == "smoke"
        updated["classification"] = label
        updated["cls"] = label
        updated["review"] = {
            "status": "reviewed",
            "label": label,
            "polygons": [
                [[float(x), float(y)] for x, y in polygon]
                for polygon in review_polygons
            ],
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        label_path = item.image_path.with_suffix(".txt")
        lines = polygons_to_yolo_lines(review_polygons, width, height)
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        _write_json(item.metadata_path, updated)
        return AlertItem(item.image_path, item.metadata_path, updated)

    @staticmethod
    def _remove_alert_files(item: AlertItem) -> None:
        for path in (item.image_path, item.metadata_path, item.image_path.with_suffix(".txt")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_dataset_yaml(root: Path) -> None:
        yaml_path = root / "dataset.yaml"
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        try:
            import yaml

            payload = {
                "path": root.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "smoke"},
            }
            yaml_path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except ImportError:
            yaml_path.write_text(
                f"path: {root.as_posix()}\ntrain: images/train\n"
                "val: images/val\ntest: images/test\nnames:\n  0: smoke\n",
                encoding="utf-8",
            )

    @staticmethod
    def _append_manifest(root: Path, record: dict[str, Any]) -> None:
        path = root / "manifest.csv"
        fields = (
            "image",
            "label",
            "split",
            "source_video",
            "frame_index",
            "timestamp_seconds",
            "label_count",
            "label_status",
        )
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow({field: record.get(field, "") for field in fields})
