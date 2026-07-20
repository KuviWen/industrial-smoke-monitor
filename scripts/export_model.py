"""Export a trained checkpoint for an offline deployment target."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=("onnx", "openvino", "engine"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    device: str | int = int(args.device) if args.device.isdigit() else args.device

    model = YOLO(args.weights)
    exported = model.export(format=args.format, imgsz=args.imgsz, half=args.half, device=device)
    print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()

