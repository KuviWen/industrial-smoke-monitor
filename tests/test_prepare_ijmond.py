import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ijmond.py"
SPEC = importlib.util.spec_from_file_location("prepare_ijmond", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_archive_root_seg_directory_does_not_mark_images_as_masks():
    image = Path("data/raw/ijmond/ijmond_seg/test/cropped/images/kooks_1_001.jpg")
    mask = Path("data/raw/ijmond/ijmond_seg/test/cropped/masks/kooks_1_001.png")
    assert MODULE._is_mask_path(image) is False
    assert MODULE._is_mask_path(mask) is True
