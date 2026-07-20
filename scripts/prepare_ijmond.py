"""Convert the IJmond raster masks into YOLO instance-segmentation labels.

The official IJmond archive already contains cropped images, raster masks, and
camera/timestamp split lists. This script uses those official lists when
available, merges the low- and high-opacity mask values into one ``smoke``
class, and writes the standard YOLO directory layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_WORDS = {
    "mask",
    "masks",
    "label",
    "labels",
    "annotation",
    "annotations",
    "seg",
    "segmentation",
}


@dataclass(frozen=True)
class Record:
    image: Path
    mask: Optional[Path]
    split: str
    source_list: str


def discover_dataset_root(input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    # Also support being called with .../test/cropped directly.
    if input_dir.is_dir() and input_dir.name.lower() == "cropped" and {
        "images",
        "masks",
    }.issubset({child.name.lower() for child in input_dir.iterdir() if child.is_dir()}):
        return input_dir.parent
    if (input_dir / "test" / "cropped").is_dir():
        return input_dir / "test"
    if (input_dir / "cropped").is_dir():
        return input_dir
    candidates = list(input_dir.rglob("cropped"))
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            return candidate.parent
    return input_dir


def _is_mask_path(path: Path) -> bool:
    """Identify mask files without treating a parent like ``ijmond_seg`` as one.

    The first implementation searched the complete path string. The official
    archive itself contains a directory named ``ijmond_seg``; splitting that
    path produced the token ``seg`` and incorrectly classified every image as
    a mask. Only the filename stem and the nearest directory names are useful
    for this decision.
    """
    stem = path.stem.lower()
    if re.search(r"(^|[_-])(mask|label|annotation|segmentation|gt)(?=$|[_-])", stem):
        return True

    parent_names = {path.parent.name.lower()}
    if path.parent.parent != path.parent:
        parent_names.add(path.parent.parent.name.lower())
    for name in parent_names:
        if name in MASK_WORDS or name.endswith(
            ("_mask", "_masks", "_label", "_labels", "_annotation", "_annotations", "_segmentation")
        ):
            return True
    return False


def _image_index(directory: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _pick(index: dict[str, list[Path]], filename: str) -> Optional[Path]:
    candidates = index.get(Path(filename).name.lower(), [])
    return candidates[0] if candidates else None


def _pick_mask(mask_index: dict[str, list[Path]], image: Path) -> Optional[Path]:
    """Match ``foo.jpg`` to the official ``foo.png`` mask."""
    target_stem = image.stem.lower()
    for candidates in mask_index.values():
        for candidate in candidates:
            if candidate.stem.lower() == target_stem:
                return candidate
    return None


def _parse_official_split_file(
    path: Path,
    image_index: dict[str, list[Path]],
    mask_index: dict[str, list[Path]],
    split: str,
) -> list[Record]:
    records: list[Record] = []
    without_mask = "without" in path.stem.lower()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [token.strip().strip('"').strip("'") for token in re.split(r"[\s,]+", line)]
        image: Optional[Path] = None
        for token in tokens:
            if Path(token).suffix.lower() in IMAGE_SUFFIXES and not _is_mask_path(Path(token)):
                image = _pick(image_index, token)
                if image:
                    break
        if image is None:
            continue
        mask = None if without_mask else _pick_mask(mask_index, image)
        records.append(Record(image=image, mask=mask, split=split, source_list=str(path)))
    return records


def records_from_official_splits(
    dataset_root: Path,
    strategy: str,
) -> list[Record]:
    cropped = dataset_root / "cropped"
    images_dir = cropped / "images"
    masks_dir = cropped / "masks"
    split_dir = cropped / "splits" / f"split_by_{strategy}"
    if not images_dir.is_dir() or not masks_dir.is_dir() or not split_dir.is_dir():
        return []

    image_index = _image_index(images_dir)
    mask_index = _image_index(masks_dir)
    records: list[Record] = []
    for path in sorted(split_dir.rglob("*.txt")):
        name = path.name.lower()
        parent = path.parent.name.lower()
        if parent == "train":
            # The archive includes 20/40/60/80/100% alternatives. Use the
            # complete 100% training list once, not all alternatives together.
            if not name.startswith("100_"):
                continue
            split = "train"
        elif name.startswith("val_"):
            split = "val"
        elif name.startswith("test_"):
            split = "test"
        else:
            continue
        records.extend(_parse_official_split_file(path, image_index, mask_index, split))

    # Deduplicate because some archives repeat list references.
    unique: dict[tuple[str, str], Record] = {}
    for record in records:
        unique[(record.split, str(record.image.resolve()))] = record
    return list(unique.values())


def _canonical_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "-mask", "_label", "-label", "_seg", "_gt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _fallback_records(dataset_root: Path, seed: int) -> list[Record]:
    cropped_images = dataset_root / "cropped" / "images"
    cropped_masks = dataset_root / "cropped" / "masks"
    images_dir = cropped_images if cropped_images.is_dir() else dataset_root
    masks_dir = cropped_masks if cropped_masks.is_dir() else dataset_root
    images = [
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not _is_mask_path(path)
    ]
    masks = [
        path for path in masks_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and _is_mask_path(path)
    ]
    mask_by_stem: dict[str, Path] = {}
    for mask in masks:
        mask_by_stem.setdefault(_canonical_stem(mask), mask)

    records: list[Record] = []
    for image in sorted(images):
        digest = int(hashlib.sha1(f"{seed}:{image}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "train" if digest < 0.8 else "val" if digest < 0.9 else "test"
        records.append(
            Record(
                image=image,
                mask=mask_by_stem.get(_canonical_stem(image)),
                split=split,
                source_list="fallback_hash_split",
            )
        )
    return records


def _load_mask(path: Optional[Path], shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read mask: {path}")
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    # IJmond uses 0 for background, 155 for low opacity, and 255 for high
    # opacity. For this alarm project both opacity levels are one smoke class.
    return (np.asarray(mask) > 0).astype(np.uint8)


def mask_to_yolo_polygons(mask: np.ndarray, min_component_area: float) -> tuple[list[str], int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    labels: list[str] = []
    total_area = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_component_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, 0.0015 * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        values: list[str] = ["0"]
        for x, y in polygon:
            values.extend((f"{max(0.0, min(1.0, x / width)):.6f}", f"{max(0.0, min(1.0, y / height)):.6f}"))
        labels.append(" ".join(values))
        total_area += int(area)
    return labels, total_area


def _clear_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    split_strategy: str,
    seed: int,
    min_component_area: float,
    overwrite: bool,
) -> dict[str, object]:
    dataset_root = discover_dataset_root(input_dir)
    if overwrite:
        _clear_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = records_from_official_splits(dataset_root, split_strategy)
    source_mode = f"official_{split_strategy}"
    if not records:
        records = _fallback_records(dataset_root, seed)
        source_mode = "fallback_hash_split"
    if not records:
        raise FileNotFoundError(
            f"No images found under {input_dir}. Expected the extracted IJmond archive. "
            f"Checked dataset root {dataset_root}; expected images under "
            f"{dataset_root / 'cropped' / 'images'} or an equivalent image directory."
        )

    manifest_rows: list[dict[str, object]] = []
    counters = {"train": {"images": 0, "positive": 0, "polygons": 0}, "val": {"images": 0, "positive": 0, "polygons": 0}, "test": {"images": 0, "positive": 0, "polygons": 0}}
    for index, record in enumerate(sorted(records, key=lambda item: (item.split, str(item.image)))):
        split = record.split if record.split in counters else "train"
        image = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {record.image}")
        mask = _load_mask(record.mask, image.shape[:2])
        labels, area = mask_to_yolo_polygons(mask, min_component_area)
        destination_stem = f"{index:06d}_{record.image.stem}"
        image_destination = output_dir / "images" / split / f"{destination_stem}{record.image.suffix.lower()}"
        label_destination = output_dir / "labels" / split / f"{destination_stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, image_destination)
        label_destination.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

        counters[split]["images"] += 1
        counters[split]["positive"] += int(bool(labels))
        counters[split]["polygons"] += len(labels)
        manifest_rows.append(
            {
                "output_image": str(image_destination.relative_to(output_dir)),
                "output_label": str(label_destination.relative_to(output_dir)),
                "source_image": str(record.image),
                "source_mask": str(record.mask) if record.mask else "",
                "split": split,
                "polygons": len(labels),
                "smoke_pixels": int(np.count_nonzero(mask)),
                "mask_area_pixels": area,
                "source_list": record.source_list,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "dataset_root": str(dataset_root),
        "source_mode": source_mode,
        "split_strategy": split_strategy,
        "seed": seed,
        "min_component_area": min_component_area,
        "counts": counters,
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Extracted IJmond archive directory")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output directory")
    parser.add_argument(
        "--split-strategy",
        choices=("camera", "timestamp", "hash"),
        default="camera",
        help="Use the dataset's official camera/timestamp split, or a deterministic fallback",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-component-area", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        split_strategy=args.split_strategy,
        seed=args.seed,
        min_component_area=args.min_component_area,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
