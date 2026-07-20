"""Merge one or more YOLO datasets into a new, collision-safe dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.dataset import IMAGE_EXTENSIONS, SPLITS, append_manifest_record, write_dataset_yaml  # noqa: E402


def _next_stem(directory: Path, stem: str) -> str:
    candidate = stem
    counter = 1
    while any((directory / f"{candidate}{extension}").exists() for extension in IMAGE_EXTENSIONS):
        candidate = f"{stem}_{counter:03d}"
        counter += 1
    return candidate


def _read_manifest(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = root / "manifest.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {(row.get("split", ""), Path(row.get("image", "")).stem): row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty; use --overwrite only for this exact target: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(output)

    copied = 0
    for source_arg in args.source:
        source = Path(source_arg).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Source dataset not found: {source}")
        manifest = _read_manifest(source)
        source_tag = source.name.replace(" ", "_")
        for split in SPLITS:
            image_dir = source / "images" / split
            label_dir = source / "labels" / split
            if not image_dir.is_dir() or not label_dir.is_dir():
                continue
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                source_label = label_dir / f"{image_path.stem}.txt"
                if not source_label.exists():
                    raise FileNotFoundError(f"Missing label for {image_path}")
                stem = _next_stem(output / "images" / split, f"{source_tag}_{image_path.stem}")
                target_image = output / "images" / split / f"{stem}{image_path.suffix.lower()}"
                target_label = output / "labels" / split / f"{stem}.txt"
                shutil.copy2(image_path, target_image)
                shutil.copy2(source_label, target_label)
                original = manifest.get((split, image_path.stem), {})
                append_manifest_record(
                    output,
                    {
                        "image": str(target_image.relative_to(output)).replace("\\", "/"),
                        "label": str(target_label.relative_to(output)).replace("\\", "/"),
                        "split": split,
                        "source_video": original.get("source_video", str(source)),
                        "frame_index": original.get("frame_index", ""),
                        "timestamp_seconds": original.get("timestamp_seconds", ""),
                        "label_count": original.get("label_count", ""),
                        "label_status": original.get("label_status", "merged"),
                    },
                )
                copied += 1

    print(f"Merged {copied} image(s) into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
