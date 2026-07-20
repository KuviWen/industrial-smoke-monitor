"""Configuration for training-independent field inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


CANONICAL_SETTINGS_PATH = Path("configs/monitor_settings.env")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _parse_roi(value: str | None) -> Optional[tuple[int, int, int, int]]:
    if value in (None, ""):
        return None
    fields = [part.strip() for part in value.split(",")]
    if len(fields) != 4:
        raise ValueError("ROI_XYXY must be four comma-separated integers: x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(field) for field in fields)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI_XYXY must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class Settings:
    project_root: Path
    settings_path: Path
    rtsp_url: str
    model_path: Path
    runtime_dir: Path
    alert_snapshots_dir: Path
    site_name: str
    camera_name: str
    device: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    min_smoke_area_ratio: float
    sample_interval_seconds: float
    positive_frames_to_alarm: int
    negative_frames_to_clear: int
    alert_repeat_seconds: float
    reconnect_seconds: float
    roi_xyxy: Optional[tuple[int, int, int, int]]
    log_level: str
    shadow_mode: bool
    allow_live_streaming: bool
    live_stream_host: str
    live_stream_port: int
    save_alert_snapshots: bool
    alert_snapshot_interval_seconds: float
    no_smoke_log_interval_seconds: float
    jpeg_quality: int
    smtp_host: str
    smtp_port: int
    smtp_security: str
    smtp_user: str
    smtp_password: str
    alert_from: str
    alert_to: tuple[str, ...]
    mail_on_recovery: bool
    smtp_timeout_seconds: float

    @classmethod
    def from_env(
        cls,
        project_root: str | Path | None = None,
        settings_path: str | Path | None = None,
    ) -> "Settings":
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        if settings_path is not None:
            selected_path = Path(settings_path).expanduser()
            if not selected_path.is_absolute():
                selected_path = root / selected_path
        else:
            canonical_path = root / CANONICAL_SETTINGS_PATH
            # Keep a useful path in diagnostics even before the user has
            # copied the example file.  External environment variables can
            # still provide values when this file is absent.
            selected_path = canonical_path
        if selected_path.is_file():
            load_dotenv(selected_path)

        recipients = tuple(
            address.strip()
            for address in os.getenv("ALERT_TO", "").split(",")
            if address.strip()
        )
        return cls(
            project_root=root,
            settings_path=selected_path.resolve(),
            rtsp_url=os.getenv("RTSP_URL", "").strip(),
            model_path=_resolve_path(root, os.getenv("MODEL_PATH", "models/best.pt")),
            runtime_dir=_resolve_path(root, os.getenv("RUNTIME_DIR", "data/runtime")),
            alert_snapshots_dir=_resolve_path(
                root, os.getenv("ALERT_SNAPSHOTS_DIR", "data/runtime/alerts")
            ),
            site_name=os.getenv("SITE_NAME", "Plant").strip(),
            camera_name=os.getenv("CAMERA_NAME", "Camera").strip(),
            device=os.getenv("DEVICE", "auto").strip(),
            imgsz=_get_int("IMGSZ", 640),
            conf_threshold=_get_float("CONF_THRESHOLD", 0.35),
            iou_threshold=_get_float("IOU_THRESHOLD", 0.50),
            min_smoke_area_ratio=_get_float("MIN_SMOKE_AREA_RATIO", 0.005),
            sample_interval_seconds=_get_float("SAMPLE_INTERVAL_SECONDS", 1.0),
            positive_frames_to_alarm=_get_int("POSITIVE_FRAMES_TO_ALARM", 5),
            negative_frames_to_clear=_get_int("NEGATIVE_FRAMES_TO_CLEAR", 30),
            alert_repeat_seconds=_get_float("ALERT_REPEAT_SECONDS", 1800.0),
            reconnect_seconds=_get_float("RECONNECT_SECONDS", 5.0),
            roi_xyxy=_parse_roi(os.getenv("ROI_XYXY")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            shadow_mode=_get_bool("SHADOW_MODE", False),
            allow_live_streaming=_get_bool("ALLOW_LIVE_STREAMING", False),
            live_stream_host=os.getenv("LIVE_STREAM_HOST", "127.0.0.1").strip(),
            live_stream_port=_get_int("LIVE_STREAM_PORT", 8765),
            save_alert_snapshots=_get_bool(
                "SAVE_ALERT_SNAPSHOTS",
                # Compatibility with the old setting name.  New templates
                # use SAVE_ALERT_SNAPSHOTS and default to retaining alerts.
                _get_bool("SAVE_POSITIVE_SNAPSHOTS", True),
            ),
            alert_snapshot_interval_seconds=_get_float(
                "ALERT_SNAPSHOT_INTERVAL_SECONDS", 1.0
            ),
            no_smoke_log_interval_seconds=_get_float(
                "NO_SMOKE_LOG_INTERVAL_SECONDS", 0.0
            ),
            jpeg_quality=max(1, min(100, _get_int("JPEG_QUALITY", 90))),
            smtp_host=os.getenv("SMTP_HOST", "").strip(),
            smtp_port=_get_int("SMTP_PORT", 25),
            smtp_security=os.getenv("SMTP_SECURITY", "none").strip().lower(),
            smtp_user=os.getenv("SMTP_USER", "").strip(),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            alert_from=os.getenv("ALERT_FROM", "").strip(),
            alert_to=recipients,
            mail_on_recovery=_get_bool("MAIL_ON_RECOVERY", True),
            smtp_timeout_seconds=_get_float("SMTP_TIMEOUT_SECONDS", 20.0),
        )

    def validate_for_monitor(self) -> None:
        """Fail early on configuration errors instead of silently running."""
        problems: list[str] = []
        if not self.rtsp_url:
            problems.append("RTSP_URL is empty")
        if not self.model_path.is_file():
            problems.append(f"MODEL_PATH does not exist: {self.model_path}")
        processed_root = (self.project_root / "data" / "processed").resolve()
        try:
            self.alert_snapshots_dir.resolve().relative_to(processed_root)
        except ValueError:
            pass
        else:
            problems.append(
                "ALERT_SNAPSHOTS_DIR must be outside data/processed so alert "
                "artifacts cannot be mixed with training data"
            )
        if not self.shadow_mode:
            if not self.smtp_host:
                problems.append("SMTP_HOST is empty")
            if not self.alert_from:
                problems.append("ALERT_FROM is empty")
            if not self.alert_to:
                problems.append("ALERT_TO has no recipients")
        if self.smtp_security not in {"none", "starttls", "ssl"}:
            problems.append("SMTP_SECURITY must be none, starttls, or ssl")
        if not 0.0 < self.conf_threshold <= 1.0:
            problems.append("CONF_THRESHOLD must be in (0, 1]")
        if not 0.0 < self.iou_threshold <= 1.0:
            problems.append("IOU_THRESHOLD must be in (0, 1]")
        if self.min_smoke_area_ratio < 0.0 or self.min_smoke_area_ratio > 1.0:
            problems.append("MIN_SMOKE_AREA_RATIO must be in [0, 1]")
        if self.sample_interval_seconds <= 0:
            problems.append("SAMPLE_INTERVAL_SECONDS must be positive")
        if self.positive_frames_to_alarm < 1:
            problems.append("POSITIVE_FRAMES_TO_ALARM must be at least 1")
        if self.negative_frames_to_clear < 1:
            problems.append("NEGATIVE_FRAMES_TO_CLEAR must be at least 1")
        if not 1 <= self.live_stream_port <= 65535:
            problems.append("LIVE_STREAM_PORT must be between 1 and 65535")
        if self.allow_live_streaming and not self.live_stream_host:
            problems.append("LIVE_STREAM_HOST is empty while live streaming is enabled")
        if self.alert_snapshot_interval_seconds <= 0:
            problems.append("ALERT_SNAPSHOT_INTERVAL_SECONDS must be positive")
        if self.no_smoke_log_interval_seconds < 0:
            problems.append("NO_SMOKE_LOG_INTERVAL_SECONDS must be >= 0")
        if problems:
            raise ValueError("Invalid monitor configuration:\n- " + "\n- ".join(problems))

    @property
    def save_positive_snapshots(self) -> bool:
        """Deprecated alias retained for integrations using the old name."""

        return self.save_alert_snapshots
