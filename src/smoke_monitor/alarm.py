"""Temporal alarm logic that suppresses single-frame false positives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AlarmEvent:
    event_type: str
    timestamp: datetime
    reason: str


class AlarmStateMachine:
    """Convert per-frame predictions into start, reminder, and clear events."""

    def __init__(
        self,
        positive_frames_to_alarm: int = 5,
        negative_frames_to_clear: int = 30,
        alert_repeat_seconds: float = 1800.0,
    ) -> None:
        self.positive_frames_to_alarm = positive_frames_to_alarm
        self.negative_frames_to_clear = negative_frames_to_clear
        self.alert_repeat_seconds = alert_repeat_seconds
        self.active = False
        self.positive_count = 0
        self.negative_count = 0
        self.last_alert_monotonic: float | None = None

    @property
    def state(self) -> str:
        return "ALARM" if self.active else "NORMAL"

    def update(self, smoke: bool, now_monotonic: float | None = None) -> list[AlarmEvent]:
        now_mono = time.monotonic() if now_monotonic is None else now_monotonic
        now = datetime.now(timezone.utc)
        events: list[AlarmEvent] = []

        if smoke:
            self.positive_count += 1
            self.negative_count = 0
            if not self.active and self.positive_count >= self.positive_frames_to_alarm:
                self.active = True
                if self._can_alert(now_mono):
                    self.last_alert_monotonic = now_mono
                    events.append(
                        AlarmEvent("smoke_started", now, "positive prediction persisted")
                    )
            elif self.active and self._can_alert(now_mono):
                self.last_alert_monotonic = now_mono
                events.append(
                    AlarmEvent("smoke_reminder", now, "smoke remains above threshold")
                )
        else:
            self.negative_count += 1
            self.positive_count = 0
            if self.active and self.negative_count >= self.negative_frames_to_clear:
                self.active = False
                self.last_alert_monotonic = None
                events.append(AlarmEvent("smoke_cleared", now, "negative prediction persisted"))

        return events

    def _can_alert(self, now_monotonic: float) -> bool:
        if self.last_alert_monotonic is None:
            return True
        return now_monotonic - self.last_alert_monotonic >= self.alert_repeat_seconds

