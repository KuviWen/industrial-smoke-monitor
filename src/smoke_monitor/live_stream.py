"""Optional browser-viewable video preview for annotated monitor frames.

The server uses Python's standard library and OpenCV only.  It keeps the
latest encoded frame in memory; it does not write a live JPG or JSON file.
The browser page consumes the frame stream, creates a MediaStream from a
canvas, and displays that MediaStream in a real ``<video>`` element.  This is
more compatible than asking browsers to play MJPEG directly as a video URL.
Older Internet Explorer and compatibility-mode browsers cannot parse the
modern JavaScript APIs used by that path and cannot play multipart JPEG as a
native ``<video>`` source.  They use an ES5 XMLHttpRequest/canvas fallback
through the in-memory ``/frame.jpg`` endpoint instead.
The default bind address is localhost.  Set ``LIVE_STREAM_HOST=0.0.0.0`` only
when the site network and firewall rules explicitly allow other clients.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    live_stream: LiveStreamServer


class LiveStreamServer:
    """Serve annotated frames for a browser video preview."""

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
        logger.info("Live browser video stream started at %s", self.url)

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
        logger.info("Live browser video stream stopped")


class _LiveRequestHandler(BaseHTTPRequestHandler):
    server: _LiveHTTPServer

    @property
    def live_stream(self) -> LiveStreamServer:
        return self.server.live_stream  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"", "/"}:
            self._send_html()
        elif path == "/stream":
            self._send_stream()
        elif path == "/frame.jpg":
            self._send_frame(parsed.query)
        elif path == "/latest.json":
            self._send_latest_json()
        elif path == "/healthz":
            self._send_health()
        else:
            self.send_error(404, "Not found")

    def _send_html(self) -> None:
        body = f"""<!doctype html>
<html lang="en"><head><meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial Smoke Monitor</title>
<style>body{{margin:0;background:#111;color:#eee;font:16px system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:16px}}
video{{display:block;width:100%;max-height:80vh;background:#000}}
a{{color:#8bd5ff}} .status{{color:#b9c6d3}}</style></head>
<body><main><h1>Industrial Smoke Monitor</h1>
<video id="live-video" autoplay muted playsinline controls
  aria-label="Live annotated smoke monitor video"></video>
<canvas id="frame-canvas" hidden></canvas>
<p id="status" class="status">Connecting to live video...</p>
<p><a href="/latest.json">Latest scores (JSON)</a> · <a href="/healthz">Health</a></p>
<script>
(function () {{
  var video = document.getElementById("live-video");
  var canvas = document.getElementById("frame-canvas");
  var context = canvas.getContext("2d");
  var status = document.getElementById("status");
  var canvasStream = null;
  var stopped = false;
  var sequence = 0;

  function setStatus(message) {{
    status.innerHTML = message;
  }}

  function appendBytes(left, right) {{
    var joined = new Uint8Array(left.length + right.length);
    joined.set(left);
    joined.set(right, left.length);
    return joined;
  }}

  function findMarker(bytes, first, second, start) {{
    var index;
    for (index = start; index + 1 < bytes.length; index += 1) {{
      if (bytes[index] === first && bytes[index + 1] === second) return index;
    }}
    return -1;
  }}

  function drawBitmap(bitmap) {{
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {{
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }}
    if (!canvasStream) {{
      canvasStream = canvas.captureStream(10);
      video.srcObject = canvasStream;
      if (video.play) {{
        video.play().then(function () {{}}, function () {{}});
      }}
    }}
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    if (bitmap.close) bitmap.close();
    setStatus("Live video connected");
  }}

  function consumeModernStream() {{
    return window.fetch("/stream", {{cache: "no-store"}}).then(function (response) {{
      if (!response.ok || !response.body) throw new Error("stream request failed");
      var reader = response.body.getReader();
      var buffer = new Uint8Array(0);

      function processFrames() {{
        var start = findMarker(buffer, 0xff, 0xd8, 0);
        var end;
        var jpeg;
        if (start < 0) {{
          if (buffer.length > 1024 * 1024) buffer = buffer.slice(-2);
          return window.Promise.resolve();
        }}
        end = findMarker(buffer, 0xff, 0xd9, start + 2);
        if (end < 0) {{
          if (start > 0) buffer = buffer.slice(start);
          return window.Promise.resolve();
        }}
        jpeg = buffer.slice(start, end + 2);
        buffer = buffer.slice(end + 2);
        return window.createImageBitmap(
          new Blob([jpeg], {{type: "image/jpeg"}})
        ).then(function (bitmap) {{
          drawBitmap(bitmap);
          return processFrames();
        }});
      }}

      function readNext() {{
        if (stopped) return window.Promise.resolve();
        return reader.read().then(function (result) {{
          if (result.done) return null;
          if (result.value) buffer = appendBytes(buffer, result.value);
          return processFrames().then(readNext);
        }});
      }}

      return readNext();
    }});
  }}

  function startModernStream() {{
    consumeModernStream().then(function () {{
      if (!stopped) {{
        setStatus("Live video reconnecting...");
        window.setTimeout(startModernStream, 1000);
      }}
    }}, function (error) {{
      if (!stopped) {{
        setStatus("Live video unavailable; retrying...");
        if (window.console && console.error) console.error(error);
        window.setTimeout(startModernStream, 1000);
      }}
    }});
  }}

  function drawLegacyFrame(blob) {{
    var image = new Image();
    var objectUrlApi = window.URL || window.webkitURL;
    var objectUrl;
    if (!objectUrlApi || !objectUrlApi.createObjectURL) {{
      setStatus("This browser cannot display the live video preview.");
      return;
    }}
    objectUrl = objectUrlApi.createObjectURL(blob);
    image.onload = function () {{
      if (canvas.width !== image.width || canvas.height !== image.height) {{
        canvas.width = image.width;
        canvas.height = image.height;
      }}
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      objectUrlApi.revokeObjectURL(objectUrl);
      setStatus("Live preview connected (compatibility mode)");
      requestLegacyFrame();
    }};
    image.onerror = function () {{
      objectUrlApi.revokeObjectURL(objectUrl);
      setStatus("Live preview frame could not be decoded; retrying...");
      window.setTimeout(requestLegacyFrame, 1000);
    }};
    image.src = objectUrl;
  }}

  function requestLegacyFrame() {{
    var request;
    if (stopped) return;
    request = new XMLHttpRequest();
    request.open("GET", "/frame.jpg?after=" + sequence + "&cache=" + new Date().getTime(), true);
    try {{ request.responseType = "blob"; }} catch (ignore) {{}}
    request.onreadystatechange = function () {{
      var nextSequence;
      if (request.readyState !== 4) return;
      if (request.status === 200) {{
        nextSequence = parseInt(request.getResponseHeader("X-Stream-Sequence"), 10);
        if (!isNaN(nextSequence)) sequence = nextSequence;
        drawLegacyFrame(request.response);
      }} else {{
        setStatus("Live preview unavailable; retrying...");
        window.setTimeout(requestLegacyFrame, 1000);
      }}
    }};
    request.onerror = function () {{
      setStatus("Live preview unavailable; retrying...");
      window.setTimeout(requestLegacyFrame, 1000);
    }};
    request.send(null);
  }}

  function start() {{
    var modernVideo = window.fetch && window.Promise && window.ReadableStream &&
      window.createImageBitmap && typeof canvas.captureStream === "function";
    if (modernVideo) {{
      startModernStream();
      return;
    }}
    video.style.display = "none";
    canvas.hidden = false;
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.maxHeight = "80vh";
    canvas.style.backgroundColor = "#000";
    setStatus("Using compatibility live preview...");
    requestLegacyFrame();
  }}

  start();
}})();
</script>
</main></body></html>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-UA-Compatible", "IE=edge")
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

    def _send_frame(self, query: str) -> None:
        """Long-poll one in-memory JPEG for legacy browser compatibility."""

        try:
            after = int(parse_qs(query).get("after", ["0"])[0])
        except (TypeError, ValueError):
            after = 0
        frame, _record, sequence = self.live_stream.wait_for_frame(
            after, timeout=10.0
        )
        if frame is None:
            self.send_error(503, "No frame available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("X-Stream-Sequence", str(sequence))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(frame)

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
