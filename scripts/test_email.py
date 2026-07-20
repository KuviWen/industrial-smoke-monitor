"""Send a test email through the configured internal SMTP relay."""

from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smoke_monitor.alarm import AlarmEvent
from smoke_monitor.detector import SmokeDetection
from smoke_monitor.emailer import EmailAlerter
from smoke_monitor.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default=None,
        help="Settings file; defaults to configs/monitor_settings.env",
    )
    args = parser.parse_args()
    settings = Settings.from_env(ROOT, args.settings)
    if not settings.smtp_host or not settings.alert_to:
        raise SystemExit(
            "Set SMTP_HOST and ALERT_TO in configs/monitor_settings.env first."
        )
    evidence = settings.runtime_dir / "email_test.jpg"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(evidence), np.zeros((100, 160, 3), dtype=np.uint8))
    detection = SmokeDetection(
        smoke=True,
        max_confidence=0.99,
        smoke_area_ratio=0.10,
        instance_count=1,
        frame_shape=(100, 160, 3),
        roi_xyxy=None,
        annotated_frame=np.zeros((100, 160, 3), dtype=np.uint8),
    )
    event = AlarmEvent("smoke_started", datetime.now(timezone.utc), "manual test")
    EmailAlerter(settings).send(event, detection, evidence)
    print("Test email sent.")


if __name__ == "__main__":
    main()
