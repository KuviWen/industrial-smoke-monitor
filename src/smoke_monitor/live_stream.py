"""Optional browser-viewable MJPEG stream for annotated monitor frames.

The server uses Python's standard library and OpenCV only.  It keeps the
latest encoded frame in memory; it does not write a live JPG or JSON file.
The default bind address is localhost.  Set ``LIVE_STREAM_HOST=0.0.0.0`` only
when the site network and firewall rules explicitly allow other clients.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    live_stream: LiveStreamServer


class LiveStreamServer:
    """Serve the latest annotated frame as an MJPEG browser stream."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        jpeg_quality: int = 85,
    ) -> None:
        self.host = host
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._record: dict[str, Any] | None = None
        self._sequence = 0
        self._closed = False
        self._started = False
        self._thread: threading.Thread | None = None
        self._server = _LiveHTTPServer((host, port), _LiveRequestHandler)
        self._server.live_stream = self

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/"

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="smoke-live-stream",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("Live MJPEG stream started at %s", self.url)

    def publish(self, frame: np.ndarray, record: dict[str, Any]) -> None:
        """Encode and publish one annotated frame without writing it to disk."""

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise IOError("OpenCV could not encode the live stream frame")
        # JSON round-tripping gives each HTTP response an immutable snapshot
        # and converts any Path-like values to strings.
        record_copy = json.loads(
            json.dumps(record, ensure_ascii=False, default=str)
        )
        with self._condition:
            if self._closed:
                return
            self._frame = encoded.tobytes()
            self._record = record_copy
            self._sequence += 1
            self._condition.notify_all()

    def wait_for_frame(
        self,
        last_sequence: int,
        timeout: float = 10.0,
    ) -> tuple[bytes | None, dict[str, Any] | None, int]:
        with self._condition:
            if self._sequence <= last_sequence and not self._closed:
                self._condition.wait(timeout=max(0.1, timeout))
            return self._frame, self._record, self._sequence

    def snapshot(self) -> tuple[bytes | None, dict[str, Any] | None, int]:
        with self._condition:
            return self._frame, self._record, self._sequence

    def close(self) -> None:
        with self._condition:
            if self._closed and not self._started:
                return
            self._closed = True
            self._condition.notify_all()
        if self._started:
            self._server.shutdown()
            if self._thread:
                self._thread.join(timeout=3.0)
        self._server.server_close()
        self._started = False
        logger.info("Live MJPEG stream stopped")


class _LiveRequestHandler(BaseHTTPRequestHandler):
    server: _LiveHTTPServer

    @property
    def live_stream(self) -> LiveStreamServer:
        return self.server.live_stream  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path in {"", "/"}:
            self._send_html()
        elif path == "/stream.mjpg":
            self._send_stream()
        elif path == "/latest.json":
            self._send_latest_json()
        elif path == "/healthz":
            self._send_health()
        else:
            self.send_error(404, "Not found")

    def _send_html(self) -> None:
        body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial Smoke Monitor</title>
<style>body{{margin:0;background:#111;color:#eee;font:16px system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:16px}} img{{display:block;max-width:100%;height:auto}}
a{{color:#8bd5ff}}</style></head>
<body><main><h1>Industrial Smoke Monitor</h1>
<img src="/stream.mjpg" alt="Live annotated smoke monitor stream">
<p><a href="/latest.json">Latest scores (JSON)</a> · <a href="/healthz">Health</a></p>
</main></body></html>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = 0
        try:
            while True:
                frame, _record, new_sequence = self.live_stream.wait_for_frame(sequence)
                if frame is None or new_sequence == sequence:
                    if self.live_stream.closed:
                        break
                    continue
                sequence = new_sequence
                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Live stream client disconnected")

    def _send_latest_json(self) -> None:
        _frame, record, sequence = self.live_stream.snapshot()
        payload = record or {"status": "waiting_for_first_frame"}
        payload = dict(payload)
        payload["stream_sequence"] = sequence
        self._send_json(payload)

    def _send_health(self) -> None:
        _frame, _record, sequence = self.live_stream.snapshot()
        self._send_json(
            {
                "status": "ok",
                "stream_sequence": sequence,
                "has_frame": sequence > 0,
            }
        )

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("Live stream HTTP: " + format, *args)
