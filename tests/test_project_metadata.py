import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _active_requirement_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_pyproject_dependencies_match_root_requirements():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(metadata["project"]["dependencies"])
    requirements = _active_requirement_lines(ROOT / "requirements.txt")
    assert dependencies == requirements
