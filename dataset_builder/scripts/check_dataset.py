"""Validate image/label pairs in a YOLO11 segmentation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.dataset import IMAGE_EXTENSIONS, SPLITS, write_dataset_yaml  # noqa: E402
from smoke_dataset_builder.yolo import validate_yolo_lines  # noqa: E402


def _images(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--no-write-yaml", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset).expanduser().resolve()
    errors: list[str] = []
    total_images = 0
    total_labels = 0
    total_negatives = 0

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"missing directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"missing directory: {label_dir}")
            continue
        image_map = _images(image_dir)
        label_map = {path.stem: path for path in label_dir.glob("*.txt")}
        total_images += len(image_map)
        total_labels += len(label_map)
        for stem, image_path in sorted(image_map.items()):
            label_path = label_dir / f"{stem}.txt"
            if not label_path.exists():
                errors.append(f"{split}: missing label for {image_path.name}")
                continue
            text = label_path.read_text(encoding="utf-8")
            if not text.strip():
                total_negatives += 1
            errors.extend(
                f"{split}/{label_path.name}: {message}"
                for message in validate_yolo_lines(text.splitlines(), args.class_count)
            )
        for stem, label_path in sorted(label_map.items()):
            if stem not in image_map:
                errors.append(f"{split}: missing image for {label_path.name}")
        print(
            f"{split}: {len(image_map)} image(s), "
            f"{len(label_map)} label(s)"
        )

    if total_images == 0:
        errors.append("dataset contains no images")
    if not args.no_write_yaml:
        write_dataset_yaml(root)

    print(f"total: {total_images} image(s), {total_labels} label(s), {total_negatives} negative sample(s)")
    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Validation passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
