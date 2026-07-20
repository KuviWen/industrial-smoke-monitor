"""Runtime evidence, alert sidecars, and JSONL records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EventStore:
    def __init__(
        self,
        runtime_dir: Path,
        jpeg_quality: int = 90,
        alert_snapshots_dir: Path | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.jpeg_quality = jpeg_quality
        self.evidence_dir = self.runtime_dir / "evidence"
        self.candidate_dir = self.runtime_dir / "candidates"
        self.alert_snapshots_dir = Path(
            alert_snapshots_dir or self.runtime_dir / "alerts"
        )
        self.records_path = self.runtime_dir / "detections.jsonl"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.alert_snapshots_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_timestamp(timestamp: datetime | None = None) -> str:
        value = timestamp or datetime.now(timezone.utc)
        return value.astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def save_evidence(
        self,
        frame: np.ndarray,
        event_type: str,
        timestamp: datetime | None = None,
    ) -> Path:
        day_dir = self.evidence_dir / (timestamp or datetime.now(timezone.utc)).astimezone().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{self._file_timestamp(timestamp)}_{event_type}.jpg"
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise IOError(f"Unable to write evidence image: {path}")
        return path

    def save_candidate(self, frame: np.ndarray, timestamp: datetime | None = None) -> Path:
        day_dir = self.candidate_dir / (timestamp or datetime.now(timezone.utc)).astimezone().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{self._file_timestamp(timestamp)}_positive.jpg"
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise IOError(f"Unable to write candidate image: {path}")
        return path

    def _write_image(self, path: Path, frame: np.ndarray) -> Path:
        """Write an image atomically so a GUI never opens a half-written file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        ok = cv2.imwrite(
            str(temporary),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise IOError(f"Unable to write image: {path}")
        os.replace(temporary, path)
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    @staticmethod
    def _available_stem(directory: Path, stem: str) -> str:
        candidate = stem
        counter = 1
        while (directory / f"{candidate}.jpg").exists() or (
            directory / f"{candidate}.json"
        ).exists():
            candidate = f"{stem}_{counter:02d}"
            counter += 1
        return candidate

    def save_alert(
        self,
        frame: np.ndarray,
        record: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> tuple[Path, Path]:
        """Save an annotated smoke alert and its reviewable JSON sidecar.

        Alert artifacts intentionally live under ``data/runtime/alerts`` (or
        the configured equivalent), never under ``data/processed``.  The
        sidecar retains the original model scores and polygons so a reviewer
        can correct the label later without needing the RTSP stream.
        """

        value = timestamp or datetime.now(timezone.utc)
        day_dir = self.alert_snapshots_dir / value.astimezone().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = self._available_stem(day_dir, f"{self._file_timestamp(value)}_smoke")
        image_path = day_dir / f"{stem}.jpg"
        metadata_path = day_dir / f"{stem}.json"
        self._write_image(image_path, frame)

        payload = dict(record)
        payload.update(
            {
                "artifact_type": "smoke_alert",
                "image_path": str(image_path),
                "metadata_path": str(metadata_path),
                "review": payload.get("review", {"status": "unreviewed"}),
            }
        )
        self._write_json(metadata_path, payload)
        return image_path, metadata_path

    def append_record(self, record: dict[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
