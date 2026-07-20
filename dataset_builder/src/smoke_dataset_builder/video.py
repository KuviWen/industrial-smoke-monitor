"""Small OpenCV helpers for reading and sampling video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".3gp")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("video operations require opencv-python") from exc
    return cv2


def read_video_info(video_path: str | Path) -> VideoInfo:
    """Open a video and return the metadata needed by the GUI/CLI."""

    cv2 = _require_cv2()
    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Video has no readable dimensions: {path}")
        return VideoInfo(path, width, height, fps if fps > 0 else 25.0, frame_count)
    finally:
        capture.release()


def iter_sampled_frames(
    video_path: str | Path,
    every_seconds: float = 1.0,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, object]]:
    """Yield ``(frame_index, timestamp_seconds, bgr_frame)`` at an interval."""

    if every_seconds <= 0:
        raise ValueError("every_seconds must be greater than zero")
    cv2 = _require_cv2()
    info = read_video_info(video_path)
    capture = cv2.VideoCapture(str(info.path))
    interval = max(1, round(info.fps * every_seconds))
    yielded = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                yield frame_index, frame_index / info.fps, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()


def safe_video_stem(video_path: str | Path) -> str:
    """Return a filesystem-safe stem suitable for generated sample names."""

    import re

    stem = Path(video_path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "video"
