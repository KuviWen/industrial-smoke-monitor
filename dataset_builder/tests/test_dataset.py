import csv
import sys
from pathlib import Path

from smoke_dataset_builder.dataset import dataset_yaml_content, safe_stem
from smoke_dataset_builder.video import SUPPORTED_VIDEO_EXTENSIONS


def test_dataset_yaml_has_parent_compatible_layout() -> None:
    config = dataset_yaml_content("./data/processed/video_yolo")
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["test"] == "images/test"
    assert config["names"] == {0: "smoke"}


def test_safe_stem_removes_windows_unfriendly_characters() -> None:
    assert safe_stem("camera 01: 2026/07/18") == "camera_01_2026_07_18"


def test_3gp_is_available_to_the_dataset_builder_video_inputs() -> None:
    assert ".3gp" in SUPPORTED_VIDEO_EXTENSIONS


def test_writer_adds_flipped_image_and_mirrored_label(monkeypatch, tmp_path: Path) -> None:
    class FakeImage:
        shape = (100, 100, 3)
        ndim = 3

    class FakeCV2:
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def imwrite(path, image, _params):
            Path(path).write_bytes(b"fake-jpeg")
            return True

        @staticmethod
        def flip(image, flip_code):
            assert flip_code == 1
            return ("flipped", image)

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    from smoke_dataset_builder.dataset import YoloDatasetWriter

    writer = YoloDatasetWriter(tmp_path / "dataset")
    original_image, original_label = writer.save_sample(
        FakeImage(),
        [[(0, 10), (99, 10), (0, 20)]],
        "train",
        "camera_f00000001",
        source_video="camera.mp4",
        frame_index=1,
        timestamp_seconds=0.5,
        generate_horizontal_flip=True,
    )

    flipped_image = original_image.with_name("camera_f00000001_flip.jpg")
    flipped_label = original_label.with_name("camera_f00000001_flip.txt")
    assert original_image.exists()
    assert original_label.exists()
    assert flipped_image.exists()
    assert flipped_label.exists()
    assert original_label.read_text(encoding="utf-8") == (
        "0 0.000000 0.100000 0.990000 0.100000 0.000000 0.200000\n"
    )
    assert flipped_label.read_text(encoding="utf-8") == (
        "0 0.990000 0.100000 0.000000 0.100000 0.990000 0.200000\n"
    )

    with (tmp_path / "dataset" / "manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["label_status"] == "reviewed"
    assert rows[1]["label_status"] == "reviewed_augmented_horizontal_flip"
    assert rows[1]["image"] == "images/train/camera_f00000001_flip.jpg"
