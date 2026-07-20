from pathlib import Path

from video_splitter.core import (
    Segment,
    VideoInfo,
    build_segments,
    format_timecode,
    parse_timecode,
    safe_stem,
    SUPPORTED_VIDEO_EXTENSIONS,
)


def make_info() -> VideoInfo:
    return VideoInfo(Path("video.mp4"), 1920, 1080, 25.0, 2500)


def test_parse_and_format_timecode() -> None:
    assert parse_timecode("00:01:02.500") == 62.5
    assert parse_timecode("01:02.5") == 62.5
    assert parse_timecode("7.25") == 7.25
    assert format_timecode(62.5) == "00:01:02.500"


def test_single_segment_is_clamped_to_video_duration() -> None:
    segments = build_segments(make_info(), mode="single", start_seconds=1, end_seconds=4)
    assert segments == [Segment(1, 1.0, 4.0)]


def test_batch_segments_include_final_short_segment() -> None:
    segments = build_segments(
        make_info(),
        mode="batch",
        start_seconds=0,
        end_seconds=4.2,
        clip_duration_seconds=2,
    )
    assert [(item.start_seconds, item.end_seconds) for item in segments] == [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 4.2),
    ]


def test_safe_stem_is_windows_friendly() -> None:
    assert safe_stem("camera 01: 2026/07/18") == "camera_01_2026_07_18"


def test_3gp_is_in_the_gui_supported_video_extension_list() -> None:
    assert ".3gp" in SUPPORTED_VIDEO_EXTENSIONS
