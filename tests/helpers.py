import shutil
import subprocess
from pathlib import Path

import pytest

from ada_eval.paths import TEST_DATA_DIR


def setup_git_repo(repo_path: Path):
    subprocess.run(["git", "init"], cwd=repo_path, check=True)


def assert_git_status(cwd: Path, *, expect_dirty: bool):
    res = subprocess.run(
        ["git", "status", "--porcelain=1"],
        cwd=cwd,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    if expect_dirty:
        assert res.stdout.strip() != ""
    else:
        if res.stdout.strip() != "":
            dbg = subprocess.run(
                ["git", "diff"],
                check=False,
                cwd=cwd,
                encoding="utf-8",
                capture_output=True,
            )
            print(dbg.stdout)
        assert res.stdout.strip() == ""


@pytest.fixture
def expanded_test_datasets(tmp_path: Path) -> Path:
    """Fixture to create a copy of the expanded test dataset within `tmp_path`."""
    dataset_path = tmp_path / "unpacked_datasets"
    expanded_test_dataset_dir = TEST_DATA_DIR / "valid_base_datasets" / "expanded"
    shutil.copytree(expanded_test_dataset_dir, dataset_path)
    return dataset_path


@pytest.fixture
def compacted_test_datasets(tmp_path: Path) -> Path:
    """Fixture to create a copy of the compacted test dataset within `tmp_path`."""
    dataset_path = tmp_path / "packed_datasets"
    compacted_test_dataset_dir = TEST_DATA_DIR / "valid_base_datasets" / "compacted"
    shutil.copytree(compacted_test_dataset_dir, dataset_path)
    return dataset_path


@pytest.fixture
def generated_test_datasets(tmp_path: Path) -> Path:
    """Fixture to create a copy of the generated test dataset within `tmp_path`."""
    dataset_path = tmp_path / "generated_datasets"
    generated_test_dataset_dir = (
        TEST_DATA_DIR / "valid_generated_datasets" / "compacted"
    )
    shutil.copytree(generated_test_dataset_dir, dataset_path)
    return dataset_path
