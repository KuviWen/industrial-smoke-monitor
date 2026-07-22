"""Create a separate ROI-cropped YOLO11 segmentation dataset.

Example::

    python dataset_builder/scripts/crop_yolo_roi.py \\
        --input data/processed/video_yolo \\
        --output data/processed/video_yolo_roi \\
        --roi 600,0,1200,500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.crop import crop_dataset  # noqa: E402
from smoke_dataset_builder.roi import format_roi, parse_roi  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Existing processed YOLO dataset; it is never modified",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output dataset directory, different from --input",
    )
    parser.add_argument(
        "--roi",
        required=True,
        help="ROI in original image coordinates: x1,y1,x2,y2",
    )
    parser.add_argument(
        "--min-mask-area",
        type=float,
        default=1.0,
        help="Drop cropped contour masks smaller than this pixel area (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in the exact output directory; input is still protected",
    )
    args = parser.parse_args()
    roi = parse_roi(args.roi)
    if roi is None:  # pragma: no cover - argparse requires a value
        raise ValueError("--roi cannot be empty")
    summary = crop_dataset(
        args.input,
        args.output,
        roi,
        overwrite=args.overwrite,
        min_mask_area=args.min_mask_area,
    )
    print(
        f"Created {summary.images} cropped image(s), {summary.polygons} polygon(s), "
        f"{summary.negatives} negative sample(s) in {summary.output}"
    )
    print(f"ROI (x1,y1,x2,y2): {format_roi(roi)}")
    print("The input dataset was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
