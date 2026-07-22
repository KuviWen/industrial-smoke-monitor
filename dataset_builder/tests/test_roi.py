from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from smoke_dataset_builder.crop import crop_dataset
from smoke_dataset_builder.roi import crop_frame, format_roi, parse_roi, validate_roi


def test_parse_roi_uses_xyxy_and_rejects_xywh() -> None:
    assert parse_roi("10,20,110,220") == (10, 20, 110, 220)
    assert format_roi((10, 20, 110, 220)) == "10,20,110,220"
    with pytest.raises(ValueError, match="positive width"):
        parse_roi("10,20,10,220")


def test_crop_frame_validates_original_image_bounds() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cropped = crop_frame(image, (10, 20, 60, 70))
    assert cropped.shape == (50, 50, 3)
    with pytest.raises(ValueError, match="outside image bounds"):
        validate_roi((0, 0, 101, 80), 100, 80)


def test_crop_dataset_keeps_source_and_clips_polygon_mask(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_roi"
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, :, 1] = 80
    source_image = image_dir / "frame_0001.jpg"
    assert cv2.imwrite(str(source_image), image)
    # The polygon crosses the left and top ROI boundaries.  Its vertices are
    # normalized against the original 120x100 image.
    (label_dir / "frame_0001.txt").write_text(
        "0 0.000000 0.000000 0.750000 0.000000 0.750000 0.800000 0.000000 0.800000\n",
        encoding="utf-8",
    )

    summary = crop_dataset(source, output, (20, 10, 80, 70))

    assert summary.images == 1
    assert summary.polygons >= 1
    assert summary.negatives == 0
    cropped = cv2.imread(
        str(output / "images" / "train" / "frame_0001.jpg"), cv2.IMREAD_COLOR
    )
    assert cropped is not None
    assert cropped.shape[:2] == (60, 60)
    assert source_image.exists()
    assert source_image.stat().st_size > 0
    output_label = output / "labels" / "train" / "frame_0001.txt"
    assert output_label.read_text(encoding="utf-8").strip().startswith("0 ")
    manifest = (output / "manifest.csv").read_text(encoding="utf-8")
    assert "roi_xyxy" in manifest
    assert "20,10,80,70" in manifest


def test_crop_dataset_never_accepts_same_source_and_output(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    (source / "images" / "train").mkdir(parents=True)
    (source / "labels" / "train").mkdir(parents=True)
    with pytest.raises(ValueError, match="different"):
        crop_dataset(source, source, (0, 0, 10, 10))
