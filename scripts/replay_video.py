"""Replay a local video through the detector for acceptance testing.

This script does not send email. It creates an annotated MP4 and prints the
number of positive samples and alarm events, so field staff can tune settings
before connecting the long-running RTSP service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smoke_monitor.alarm import AlarmStateMachine  # noqa: E402
from smoke_monitor.detector import SmokeDetector  # noqa: E402
from smoke_monitor.settings import _parse_roi  # noqa: E402


def parse_roi(value: str | None):
    return _parse_roi(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="data/runtime/replay_annotated.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--area", type=float, default=0.005)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--positive-frames", type=int, default=5)
    parser.add_argument("--negative-frames", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--roi", default="")
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    detector = SmokeDetector(
        model_path=Path(args.weights),
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        min_smoke_area_ratio=args.area,
        device=str(device),
        roi_xyxy=parse_roi(args.roi),
    )
    source = cv2.VideoCapture(args.source)
    if not source.isOpened():
        raise SystemExit(f"Unable to open video: {args.source}")
    fps = source.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every = max(1, int(round(fps * args.sample_seconds)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        source.release()
        raise SystemExit(f"Unable to create output video: {output}")

    machine = AlarmStateMachine(args.positive_frames, args.negative_frames, 10**9)
    frame_number = 0
    positive_samples = 0
    events: list[str] = []
    last_annotated = None
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            if frame_number % sample_every == 0:
                detection = detector.detect(frame)
                last_annotated = detection.annotated_frame
                positive_samples += int(detection.smoke)
                events.extend(event.event_type for event in machine.update(detection.smoke, frame_number))
            writer.write(last_annotated if last_annotated is not None else frame)
            frame_number += 1
    finally:
        source.release()
        writer.release()
    print(f"Frames: {frame_number}")
    print(f"Positive samples: {positive_samples}")
    print(f"Events: {events}")
    print(f"Annotated video: {output}")


if __name__ == "__main__":
    main()
