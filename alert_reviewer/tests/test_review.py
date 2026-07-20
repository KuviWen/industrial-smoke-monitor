import json
from pathlib import Path

import cv2
import numpy as np

from alert_reviewer.review import AlertReviewService, polygon_to_yolo_line


def test_polygon_to_yolo_line_normalizes_pixels() -> None:
    assert polygon_to_yolo_line([(0, 0), (50, 0), (50, 100)], 100, 200) == (
        "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"
    )


def test_service_can_find_and_keep_reviewed_alert(tmp_path: Path) -> None:
    alert_dir = tmp_path / "alerts" / "2026-07-20"
    alert_dir.mkdir(parents=True)
    image_path = alert_dir / "sample_smoke.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    (alert_dir / "sample_smoke.json").write_text(
        json.dumps(
            {
                "artifact_type": "smoke_alert",
                "image_path": str(image_path),
                "classification": "smoke",
                "instances": [
                    {"polygon": [[10, 10], [40, 10], [40, 30]]}
                ],
            }
        ),
        encoding="utf-8",
    )

    service = AlertReviewService(tmp_path / "alerts")
    items = service.list_alerts()
    assert len(items) == 1
    reviewed = service.keep_in_place(
        items[0], "no_smoke", [[(10, 10), (40, 10), (40, 30)]]
    )
    assert reviewed.metadata["review"]["label"] == "no_smoke"
    assert (image_path.with_suffix(".txt")).read_text(encoding="utf-8") == ""


def test_service_moves_reviewed_copy_to_parent_layout(tmp_path: Path) -> None:
    alert_dir = tmp_path / "alerts"
    alert_dir.mkdir()
    image_path = alert_dir / "sample.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    (alert_dir / "sample.json").write_text(
        json.dumps({"image_path": str(image_path), "classification": "smoke"}),
        encoding="utf-8",
    )
    service = AlertReviewService(alert_dir)
    item = service.list_alerts()[0]
    image_out, label_out = service.move_to_dataset(
        item,
        "smoke",
        [[(10, 10), (40, 10), (40, 30)]],
        tmp_path / "dataset",
    )
    assert image_out == tmp_path / "dataset/images/train/alert_sample.jpg"
    assert label_out.is_file()
    assert not image_path.exists()
    assert not (alert_dir / "sample.json").exists()
