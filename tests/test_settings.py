from pathlib import Path

import pytest

from smoke_monitor.settings import _parse_roi, Settings


def test_parse_roi():
    assert _parse_roi("1,2,300,400") == (1, 2, 300, 400)
    assert _parse_roi("") is None


def test_parse_roi_rejects_invalid_order():
    with pytest.raises(ValueError):
        _parse_roi("300,2,1,400")


def test_invalid_monitor_configuration_reports_missing_fields(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("ALERT_FROM", raising=False)
    monkeypatch.delenv("ALERT_TO", raising=False)
    settings = Settings.from_env(tmp_path)
    with pytest.raises(ValueError) as error:
        settings.validate_for_monitor()
    assert "RTSP_URL is empty" in str(error.value)


def test_non_hidden_settings_file_supports_shadow_mode_without_smtp(
    tmp_path: Path, monkeypatch
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")
    settings_file = tmp_path / "configs" / "monitor_settings.env"
    settings_file.parent.mkdir()
    settings_file.write_text(
        "RTSP_URL=rtsp://camera\n"
        "MODEL_PATH=model.pt\n"
        "SHADOW_MODE=true\n"
        "NO_SMOKE_LOG_INTERVAL_SECONDS=60\n"
        "ALLOW_LIVE_STREAMING=true\n"
        "LIVE_STREAM_PORT=9876\n"
        "SAVE_ALERT_SNAPSHOTS=true\n",
        encoding="utf-8",
    )
    for name in (
        "RTSP_URL",
        "MODEL_PATH",
        "SHADOW_MODE",
        "NO_SMOKE_LOG_INTERVAL_SECONDS",
        "ALLOW_LIVE_STREAMING",
        "LIVE_STREAM_PORT",
        "SAVE_ALERT_SNAPSHOTS",
        "SMTP_HOST",
        "ALERT_FROM",
        "ALERT_TO",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(tmp_path, settings_file)

    assert settings.settings_path == settings_file.resolve()
    assert settings.shadow_mode is True
    assert settings.no_smoke_log_interval_seconds == 60
    assert settings.allow_live_streaming is True
    assert settings.live_stream_port == 9876
    settings.validate_for_monitor()


def test_negative_no_smoke_log_interval_is_rejected(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")
    monkeypatch.setenv("RTSP_URL", "rtsp://camera")
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("NO_SMOKE_LOG_INTERVAL_SECONDS", "-1")

    settings = Settings.from_env(tmp_path)
    with pytest.raises(ValueError, match="NO_SMOKE_LOG_INTERVAL_SECONDS"):
        settings.validate_for_monitor()
