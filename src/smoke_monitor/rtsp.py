"""Small reconnecting RTSP reader using OpenCV's FFmpeg backend."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPSource:
    def __init__(self, url: str, reconnect_seconds: float = 5.0) -> None:
        self.url = url
        self.reconnect_seconds = reconnect_seconds
        self.capture: Optional[cv2.VideoCapture] = None
        self._last_reconnect = 0.0

    def open(self) -> bool:
        now = time.monotonic()
        if now - self._last_reconnect < self.reconnect_seconds:
            return False
        self._last_reconnect = now
        self.close()
        # This must be set before VideoCapture is created.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        try:
            self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except Exception:
            self.capture = cv2.VideoCapture(self.url)
        if not self.capture.isOpened():
            logger.error("Unable to open RTSP stream")
            self.close()
            return False
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        logger.info("RTSP stream connected")
        return True

    def read(self) -> Optional[np.ndarray]:
        if self.capture is None or not self.capture.isOpened():
            self.open()
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            logger.warning("RTSP frame read failed; reconnecting")
            self.close()
            return None
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

