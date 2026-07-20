"""SMTP email alerting. Credentials are read only from environment settings."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

from .alarm import AlarmEvent
from .detector import SmokeDetection
from .settings import Settings

logger = logging.getLogger(__name__)


class EmailAlerter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send(
        self,
        event: AlarmEvent,
        detection: SmokeDetection,
        evidence_path: Path,
    ) -> None:
        subject_map = {
            "smoke_started": "煙囪冒煙警報",
            "smoke_reminder": "煙囪冒煙持續警報",
            "smoke_cleared": "煙囪冒煙解除通知",
        }
        message = EmailMessage()
        message["Subject"] = (
            f"[{self.settings.site_name}] "
            f"{subject_map.get(event.event_type, event.event_type)} - "
            f"{self.settings.camera_name}"
        )
        message["From"] = self.settings.alert_from
        message["To"] = ", ".join(self.settings.alert_to)
        message.set_content(
            "工業煙囪影像監測通知\n\n"
            f"廠區：{self.settings.site_name}\n"
            f"攝影機：{self.settings.camera_name}\n"
            f"事件：{event.event_type}\n"
            f"原因：{event.reason}\n"
            f"時間（UTC）：{event.timestamp.isoformat()}\n"
            f"分類：{'smoke' if detection.smoke else 'no_smoke'}\n"
            f"模型信心度：{detection.max_confidence:.4f}\n"
            f"煙霧面積比例：{detection.smoke_area_ratio:.6f}\n"
            f"偵測實例數：{detection.instance_count}\n"
            f"ROI：{detection.roi_xyxy or 'full frame'}\n"
            f"證據影像：{evidence_path}\n\n"
            "請依現場安全程序確認，不要只依賴此影像通知作為唯一安全聯鎖。\n"
        )
        if evidence_path.is_file():
            message.add_attachment(
                evidence_path.read_bytes(),
                maintype="image",
                subtype="jpeg",
                filename=evidence_path.name,
            )

        self._send_message(message)

    def _send_message(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        if self.settings.smtp_security == "ssl":
            with smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=self.settings.smtp_timeout_seconds,
                context=context,
            ) as smtp:
                self._authenticate_and_send(smtp, message)
        else:
            with smtplib.SMTP(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=self.settings.smtp_timeout_seconds,
            ) as smtp:
                if self.settings.smtp_security == "starttls":
                    smtp.starttls(context=context)
                self._authenticate_and_send(smtp, message)

    def _authenticate_and_send(self, smtp: smtplib.SMTP, message: EmailMessage) -> None:
        if self.settings.smtp_user:
            smtp.login(self.settings.smtp_user, self.settings.smtp_password)
        smtp.send_message(message)
