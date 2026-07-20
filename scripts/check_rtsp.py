"""Read a short RTSP sample and save one frame for connectivity testing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smoke_monitor.rtsp import RTSPSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="Defaults to RTSP_URL in configs/monitor_settings.env",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="Settings file; defaults to configs/monitor_settings.env",
    )
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("data/runtime/rtsp_test.jpg"))
    args = parser.parse_args()

    if args.url is None:
        from smoke_monitor.settings import Settings

        args.url = Settings.from_env(ROOT, args.settings).rtsp_url
    if not args.url:
        raise SystemExit("No RTSP URL. Set RTSP_URL or pass --url.")

    source = RTSPSource(args.url, reconnect_seconds=1.0)
    deadline = time.monotonic() + args.seconds
    frames = 0
    try:
        while time.monotonic() < deadline:
            frame = source.read()
            if frame is None:
                time.sleep(0.2)
                continue
            frames += 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), frame)
    finally:
        source.close()
    print(f"Received {frames} frames. Last frame: {args.output}")
    if frames == 0:
        raise SystemExit("RTSP test failed: no frames received.")


if __name__ == "__main__":
    main()
