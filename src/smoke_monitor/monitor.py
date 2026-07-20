"""Long-running field monitor: RTSP -> YOLO11-seg -> temporal alarm -> email."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from threading import Event

from .alarm import AlarmStateMachine
from .detector import SmokeDetector
from .emailer import EmailAlerter
from .live_stream import LiveStreamServer
from .rtsp import RTSPSource
from .settings import Settings
from .storage import EventStore

logger = logging.getLogger(__name__)


def run_monitor(settings: Settings) -> None:
    settings.validate_for_monitor()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)

    detector = SmokeDetector(
        model_path=settings.model_path,
        imgsz=settings.imgsz,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        min_smoke_area_ratio=settings.min_smoke_area_ratio,
        device=settings.device,
        roi_xyxy=settings.roi_xyxy,
    )
    source = RTSPSource(settings.rtsp_url, settings.reconnect_seconds)
    store = EventStore(
        settings.runtime_dir,
        settings.jpeg_quality,
        alert_snapshots_dir=settings.alert_snapshots_dir,
    )
    alerter = EmailAlerter(settings)
    alarm = AlarmStateMachine(
        positive_frames_to_alarm=settings.positive_frames_to_alarm,
        negative_frames_to_clear=settings.negative_frames_to_clear,
        alert_repeat_seconds=settings.alert_repeat_seconds,
    )
    stop_event = Event()
    live_stream: LiveStreamServer | None = None

    def stop_handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; stopping monitor", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, stop_handler)
        except (ValueError, OSError):
            # Signal registration can be unavailable in a non-main thread.
            pass

    last_inference = 0.0
    last_alert_snapshot = float("-inf")
    read_failures = 0
    logger.info(
        "Monitor started: site=%s camera=%s interval=%.2fs roi=%s shadow_mode=%s live_streaming=%s",
        settings.site_name,
        settings.camera_name,
        settings.sample_interval_seconds,
        settings.roi_xyxy or "full frame",
        settings.shadow_mode,
        settings.allow_live_streaming,
    )

    try:
        if settings.allow_live_streaming:
            live_stream = LiveStreamServer(
                host=settings.live_stream_host,
                port=settings.live_stream_port,
                jpeg_quality=settings.jpeg_quality,
            )
            live_stream.start()

        while not stop_event.is_set():
            frame = source.read()
            if frame is None:
                read_failures += 1
                stop_event.wait(min(1.0, settings.reconnect_seconds))
                continue
            read_failures = 0

            now_monotonic = time.monotonic()
            if now_monotonic - last_inference < settings.sample_interval_seconds:
                continue
            last_inference = now_monotonic

            try:
                detection = detector.detect(frame)
            except Exception:
                logger.exception("Inference failed; continuing with next frame")
                continue

            events = alarm.update(detection.smoke, now_monotonic)
            timestamp = datetime.now(timezone.utc)
            record = {
                "timestamp_utc": timestamp.isoformat(),
                "site_name": settings.site_name,
                "camera_name": settings.camera_name,
                "alarm_state": alarm.state,
                "read_failures": read_failures,
                "shadow_mode": settings.shadow_mode,
                "settings_path": str(settings.settings_path),
                **detection.to_record(),
                "events": [event.event_type for event in events],
            }

            if (
                settings.save_alert_snapshots
                and detection.smoke
                and now_monotonic - last_alert_snapshot
                >= settings.alert_snapshot_interval_seconds
            ):
                try:
                    alert_image_path, alert_metadata_path = store.save_alert(
                        detection.annotated_frame,
                        record,
                        timestamp,
                    )
                    record["alert_image_path"] = str(alert_image_path)
                    record["alert_metadata_path"] = str(alert_metadata_path)
                    last_alert_snapshot = now_monotonic
                except Exception:
                    logger.exception("Unable to save smoke alert artifact")

            if live_stream is not None:
                record["live_stream_url"] = live_stream.url
                try:
                    live_stream.publish(detection.annotated_frame, record)
                except Exception:
                    # A browser client must never be able to stop inference.
                    logger.exception("Unable to publish live stream frame")

            if events:
                # One image is attached to all events generated at this sample.
                try:
                    evidence_path = store.save_evidence(
                        detection.annotated_frame,
                        events[0].event_type,
                        timestamp,
                    )
                    record["evidence_path"] = str(evidence_path)
                except Exception:
                    logger.exception("Unable to save evidence image")
                    evidence_path = settings.runtime_dir / "evidence_unavailable.jpg"

                for event in events:
                    if event.event_type == "smoke_cleared" and not settings.mail_on_recovery:
                        record.setdefault("email_events", []).append(
                            {"event": event.event_type, "status": "disabled"}
                        )
                        continue
                    if settings.shadow_mode:
                        logger.warning(
                            "Shadow mode: suppressing email for event %s",
                            event.event_type,
                        )
                        record.setdefault("email_events", []).append(
                            {"event": event.event_type, "status": "suppressed_shadow_mode"}
                        )
                        continue
                    try:
                        alerter.send(event, detection, evidence_path)
                        logger.warning("Email sent for event %s", event.event_type)
                        record.setdefault("email_events", []).append(
                            {"event": event.event_type, "status": "sent"}
                        )
                    except Exception as exc:
                        # Monitoring must continue even if the mail relay is down.
                        logger.exception("Email send failed for event %s", event.event_type)
                        record.setdefault("email_events", []).append(
                            {"event": event.event_type, "status": "failed", "error": str(exc)}
                        )

            store.append_record(record)

    finally:
        source.close()
        if live_stream is not None:
            live_stream.close()
        logger.info("Monitor stopped")
