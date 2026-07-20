"""Start the long-running field monitor."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smoke_monitor.logging_utils import setup_logging  # noqa: E402
from smoke_monitor.monitor import run_monitor  # noqa: E402
from smoke_monitor.settings import Settings  # noqa: E402


def self_test(settings_path: str | Path | None = None) -> int:
    """Load the configured model without connecting to RTSP or SMTP."""
    try:
        settings = Settings.from_env(ROOT, settings_path)
        if not settings.model_path.is_file():
            print(f"SELF-TEST FAILED: model not found: {settings.model_path}", file=sys.stderr)
            return 2

        import torch
        from ultralytics import YOLO

        model = YOLO(str(settings.model_path))
        print(f"model: {settings.model_path}")
        print(f"classes: {getattr(model, 'names', {})}")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print("SELF-TEST OK")
        return 0
    except Exception as exc:  # pragma: no cover - exercised by the runtime entrypoint
        print(f"SELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1


def check_rtsp(seconds: float, settings_path: str | Path | None = None) -> int:
    """Read a short RTSP sample without loading the YOLO model."""
    import cv2

    from smoke_monitor.rtsp import RTSPSource

    settings = Settings.from_env(ROOT, settings_path)
    if not settings.rtsp_url:
        print("RTSP CHECK FAILED: set RTSP_URL in configs/monitor_settings.env", file=sys.stderr)
        return 2

    setup_logging(settings.runtime_dir / "logs", settings.log_level)
    output = settings.runtime_dir / "rtsp_test.jpg"
    source = RTSPSource(settings.rtsp_url, reconnect_seconds=1.0)
    deadline = time.monotonic() + max(0.1, seconds)
    frames = 0
    try:
        while time.monotonic() < deadline:
            frame = source.read()
            if frame is None:
                time.sleep(0.2)
                continue
            frames += 1
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), frame)
    finally:
        source.close()

    print(f"Received {frames} frames. Last frame: {output}")
    if frames == 0:
        print("RTSP CHECK FAILED: no frames received", file=sys.stderr)
        return 1
    print("RTSP CHECK OK")
    return 0


def test_email(settings_path: str | Path | None = None) -> int:
    """Send a controlled test message without starting the monitor loop."""
    import cv2
    import numpy as np

    from smoke_monitor.alarm import AlarmEvent
    from smoke_monitor.detector import SmokeDetection
    from smoke_monitor.emailer import EmailAlerter

    settings = Settings.from_env(ROOT, settings_path)
    if not settings.smtp_host or not settings.alert_from or not settings.alert_to:
        print(
            "EMAIL TEST FAILED: set SMTP_HOST, ALERT_FROM, and ALERT_TO in "
            "configs/monitor_settings.env",
            file=sys.stderr,
        )
        return 2

    evidence = settings.runtime_dir / "email_test.jpg"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((100, 160, 3), dtype=np.uint8)
    if not cv2.imwrite(str(evidence), blank):
        print(f"EMAIL TEST FAILED: cannot write {evidence}", file=sys.stderr)
        return 1

    detection = SmokeDetection(
        smoke=True,
        max_confidence=0.99,
        smoke_area_ratio=0.10,
        instance_count=1,
        frame_shape=(100, 160, 3),
        roi_xyxy=None,
        annotated_frame=blank,
    )
    event = AlarmEvent("smoke_started", datetime.now(timezone.utc), "manual test")
    try:
        EmailAlerter(settings).send(event, detection, evidence)
    except Exception as exc:
        print(f"EMAIL TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print("EMAIL TEST OK: test email sent")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Load the model only")
    parser.add_argument("--check-rtsp", action="store_true", help="Read a short RTSP sample")
    parser.add_argument("--test-email", action="store_true", help="Send one test email")
    parser.add_argument("--seconds", type=float, default=15.0, help="RTSP test duration")
    parser.add_argument(
        "--settings",
        default=None,
        help="Settings file; defaults to configs/monitor_settings.env",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        raise SystemExit(self_test(args.settings))
    if args.check_rtsp:
        raise SystemExit(check_rtsp(args.seconds, args.settings))
    if args.test_email:
        raise SystemExit(test_email(args.settings))

    settings = Settings.from_env(ROOT, args.settings)
    setup_logging(settings.runtime_dir / "logs", settings.log_level)
    run_monitor(settings)


if __name__ == "__main__":
    main()
