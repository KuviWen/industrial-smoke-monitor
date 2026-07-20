"""Fine-tune a pretrained YOLO11 segmentation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | None) -> Path:
    """Resolve a training output root relative to this project, not cwd.

    The command is often launched while inspecting a previous run under
    ``runs/segment``.  Passing ``runs/train`` directly to Ultralytics in that
    situation creates ``runs/segment/runs/train``.  Anchoring relative paths
    to the repository root keeps the documented output location stable.
    """

    project = Path(value or "runs/train").expanduser()
    if not project.is_absolute():
        project = PROJECT_ROOT / project
    return project.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/ijmond.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="yolo11n-seg.pt or yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0, 1, cpu, or auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--project",
        default=None,
        help="output root; relative paths are anchored to the project root (default: runs/train)",
    )
    parser.add_argument("--name", default="smoke_yolo11n")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    device: str | int = args.device
    if args.device.isdigit():
        device = int(args.device)

    from ultralytics import YOLO

    project = resolve_project_path(args.project)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(project),
        name=args.name,
        seed=args.seed,
        cache=args.cache,
        pretrained=True,
        plots=True,
    )
    print(f"Training complete. Inspect: {project / args.name}")


if __name__ == "__main__":
    main()
