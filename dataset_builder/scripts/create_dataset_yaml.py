"""Create the dataset.yaml expected by Ultralytics YOLO11."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.dataset import write_dataset_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="YOLO dataset root")
    parser.add_argument("--class-name", default="smoke")
    args = parser.parse_args()
    path = write_dataset_yaml(args.dataset, args.class_name)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
