from pathlib import Path

from scripts.train import PROJECT_ROOT, resolve_project_path


def test_default_training_project_is_root_runs_train() -> None:
    assert resolve_project_path(None) == (PROJECT_ROOT / "runs" / "train").resolve()


def test_relative_training_project_is_anchored_to_project_root() -> None:
    assert resolve_project_path("runs/train") == (PROJECT_ROOT / "runs" / "train").resolve()


def test_absolute_training_project_is_preserved() -> None:
    absolute = Path(PROJECT_ROOT).parent / "isolated-runs" / "train"
    assert resolve_project_path(str(absolute)) == absolute.resolve()
