"""OpenCV-based video metadata, timecode, and segment writing utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable


ProgressCallback = Callable[[float, str], None]
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


@dataclass(frozen=True)
class Segment:
    """A half-open time interval: [start_seconds, end_seconds)."""

    number: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("video splitting requires opencv-python") from exc
    return cv2


def safe_stem(value: str) -> str:
    """Make a video stem safe for Windows filenames."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "video"


def parse_timecode(value: str | int | float) -> float:
    """Parse seconds or ``HH:MM:SS(.mmm)`` / ``MM:SS(.mmm)`` text."""

    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("時間不可為空白")
        if ":" not in text:
            try:
                seconds = float(text)
            except ValueError as exc:
                raise ValueError(f"無法解析時間: {value!r}") from exc
        else:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("時間格式應為秒數、MM:SS 或 HH:MM:SS")
            try:
                numbers = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"無法解析時間: {value!r}") from exc
            if len(numbers) == 2:
                minutes, second_part = numbers
                if minutes < 0 or not 0 <= second_part < 60:
                    raise ValueError("MM:SS 的秒數必須介於 0 到 60 以下")
                seconds = minutes * 60 + second_part
            else:
                hours, minutes, second_part = numbers
                if hours < 0 or not 0 <= minutes < 60 or not 0 <= second_part < 60:
                    raise ValueError("HH:MM:SS 的分鐘與秒數必須介於 0 到 60 以下")
                seconds = hours * 3600 + minutes * 60 + second_part
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("時間必須是大於等於 0 的有限數字")
    return seconds


def format_timecode(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` for a GUI field."""

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite non-negative number")
    milliseconds_total = round(seconds * 1000)
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def read_video_info(video_path: str | Path) -> VideoInfo:
    """Open a local video and return its dimensions, FPS, and frame count."""

    cv2 = _require_cv2()
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到影片: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV 無法開啟影片: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"影片沒有可讀取的影像尺寸: {path}")
        if fps <= 0:
            fps = 25.0
        if frame_count <= 0:
            raise RuntimeError(f"影片沒有可讀取的影格數: {path}")
        return VideoInfo(path, width, height, fps, frame_count)
    finally:
        capture.release()


def build_segments(
    info: VideoInfo,
    *,
    mode: str = "single",
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    clip_duration_seconds: float = 60.0,
    step_seconds: float | None = None,
) -> list[Segment]:
    """Build one segment or consecutive batch segments within a video."""

    duration = info.duration_seconds
    start = float(start_seconds)
    end = duration if end_seconds is None else float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("起訖時間必須是有限數字")
    if start < 0 or end > duration + (1.0 / info.fps) or start >= end:
        raise ValueError(
            f"切割範圍無效：影片長度 {format_timecode(duration)}，"
            f"目前為 {format_timecode(start)} 到 {format_timecode(end)}"
        )
    start = max(0.0, start)
    end = min(duration, end)
    if mode == "single":
        return [Segment(1, start, end)]
    if mode != "batch":
        raise ValueError("mode 必須是 single 或 batch")
    clip_duration = float(clip_duration_seconds)
    step = clip_duration if step_seconds is None else float(step_seconds)
    if not math.isfinite(clip_duration) or clip_duration <= 0:
        raise ValueError("每段長度必須大於 0")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("批次切割間隔必須大於 0")

    segments: list[Segment] = []
    current = start
    number = 1
    while current < end - (1.0 / info.fps) / 2:
        segment_end = min(current + clip_duration, end)
        if segment_end > current:
            segments.append(Segment(number, current, segment_end))
            number += 1
        next_current = current + step
        if next_current <= current:
            raise ValueError("批次切割間隔造成無限迴圈")
        current = next_current
    return segments


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _writer_for(cv2, path: Path, info: VideoInfo, codec: str):
    """Open a writer, trying a small set of codecs for common Windows builds."""

    codecs = [codec]
    for fallback in ("mp4v", "avc1"):
        if fallback not in codecs:
            codecs.append(fallback)
    for candidate in codecs:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*candidate),
            info.fps,
            (info.width, info.height),
        )
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise RuntimeError(
        "OpenCV 無法建立輸出影片。請確認輸出資料夾可寫入，"
        "或在本機安裝包含 mp4v 編碼器的 OpenCV。"
    )


def split_video(
    video_path: str | Path,
    output_dir: str | Path,
    segments: Iterable[Segment],
    *,
    prefix: str | None = None,
    overwrite: bool = False,
    extension: str = ".mp4",
    codec: str = "mp4v",
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> list[Path]:
    """Re-encode requested segments to independent video files.

    OpenCV writes video streams only; audio tracks are not copied.  A
    callback receives ``(0..1, message)`` from the worker thread.
    """

    cv2 = _require_cv2()
    info = read_video_info(video_path)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    segment_list = list(segments)
    if not segment_list:
        raise ValueError("至少需要一段切割範圍")
    base = safe_stem(prefix or info.path.stem)
    suffix = extension if extension.startswith(".") else f".{extension}"
    expected_frames = sum(
        max(1, min(info.frame_count, math.ceil(segment.end_seconds * info.fps)) - max(0, math.floor(segment.start_seconds * info.fps)))
        for segment in segment_list
    )
    completed_frames = 0
    results: list[Path] = []

    for segment_index, segment in enumerate(segment_list, start=1):
        if cancel_event is not None and cancel_event.is_set():
            break
        start_frame = max(0, math.floor(segment.start_seconds * info.fps))
        end_frame = min(info.frame_count, math.ceil(segment.end_seconds * info.fps))
        if end_frame <= start_frame:
            continue
        requested_path = output_root / f"{base}_{segment_index:04d}{suffix}"
        output_path = requested_path if overwrite else _unique_output_path(requested_path)
        capture = cv2.VideoCapture(str(info.path))
        writer = None
        frames_written = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"無法重新開啟影片: {info.path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer, codec_used = _writer_for(cv2, output_path, info, codec)
            frame_index = start_frame
            while frame_index < end_frame:
                if cancel_event is not None and cancel_event.is_set():
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                frames_written += 1
                frame_index += 1
                completed_frames += 1
                if progress_callback:
                    ratio = completed_frames / max(1, expected_frames)
                    progress_callback(
                        min(1.0, ratio),
                        f"第 {segment_index}/{len(segment_list)} 段，"
                        f"{frames_written} frames，codec={codec_used}",
                    )
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if frames_written > 0:
            results.append(output_path)
        if cancel_event is not None and cancel_event.is_set():
            break

    if progress_callback and results and not (cancel_event and cancel_event.is_set()):
        progress_callback(1.0, f"完成，共輸出 {len(results)} 個檔案")
    return results
