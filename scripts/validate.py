"""Validate a trained YOLO11 segmentation checkpoint."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/ijmond.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="smoke_yolo11_test")
    args = parser.parse_args()
    device: str | int = int(args.device) if args.device.isdigit() else args.device

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        plots=True,
    )
    print(f"box mAP50-95:  {metrics.box.map:.4f}")
    print(f"mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"box precision: {metrics.box.mp:.4f}")
    print(f"box recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

