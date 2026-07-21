import json
from urllib.request import urlopen

import numpy as np

from smoke_monitor.live_stream import LiveStreamServer


def test_live_stream_serves_annotated_frame_and_scores():
    server = LiveStreamServer(host="127.0.0.1", port=0, jpeg_quality=80)
    server.start()
    try:
        server.publish(
            np.zeros((24, 32, 3), dtype=np.uint8),
            {"classification": "smoke", "conf": 0.91, "roi_xyxy": [0, 0, 32, 24]},
        )

        with urlopen(server.url + "healthz", timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["has_frame"] is True
        assert health["stream_sequence"] == 1

        with urlopen(server.url + "latest.json", timeout=3) as response:
            latest = json.loads(response.read().decode("utf-8"))
        assert latest["classification"] == "smoke"
        assert latest["conf"] == 0.91

        with urlopen(server.url + "frame.jpg?after=0", timeout=3) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.headers["X-Stream-Sequence"] == "1"
            assert response.read(2) == b"\xff\xd8"

        with urlopen(server.url, timeout=3) as response:
            assert response.headers["X-UA-Compatible"] == "IE=edge"
            page = response.read().decode("utf-8")
        assert "<video" in page
        assert 'http-equiv="X-UA-Compatible"' in page
        assert 'fetch("/stream"' in page
        assert "/frame.jpg?after=" in page
        assert "async function" not in page
        assert "=>" not in page
        assert '<img src="/stream' not in page

        with urlopen(server.url + "stream", timeout=3) as response:
            assert response.headers["Content-Type"].startswith(
                "multipart/x-mixed-replace"
            )
            assert response.read(2) == b"--"
    finally:
        server.close()
