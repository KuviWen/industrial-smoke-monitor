"""Extract review candidates from a video into a temporary staging folder."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

CHILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHILD_ROOT / "src"))

from smoke_dataset_builder.video import (  # noqa: E402
    SUPPORTED_VIDEO_EXTENSIONS,
    iter_sampled_frames,
    safe_video_stem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    extensions = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
    parser.add_argument("--video", required=True, help=f"Input video file ({extensions})")
    parser.add_argument(
        "--output",
        default="dataset_builder/data/staging",
        help="Temporary staging directory; not the final YOLO dataset",
    )
    parser.add_argument("--every-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    video_output = Path(args.output).expanduser().resolve() / safe_video_stem(video_path)
    frames_dir = video_output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = video_output / "manifest.csv"
    records: list[dict[str, object]] = []

    for frame_index, timestamp, frame in iter_sampled_frames(
        video_path, args.every_seconds, args.max_frames
    ):
        image_path = frames_dir / f"frame_{frame_index:08d}.jpg"
        if image_path.exists() and not args.overwrite:
            continue
        if not cv2.imwrite(
            str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        ):
            raise IOError(f"Could not write frame: {image_path}")
        records.append(
            {
                "image": str(image_path.relative_to(video_output)).replace("\\", "/"),
                "source_video": str(video_path),
                "frame_index": frame_index,
                "timestamp_seconds": f"{timestamp:.3f}",
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image", "source_video", "frame_index", "timestamp_seconds"),
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Staged {len(records)} frame(s) in {video_output}")
    print("These frames are candidates only; review and save them with the GUI.")


if __name__ == "__main__":
    main()
