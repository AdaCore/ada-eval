import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from ada_eval.datasets import Dataset, Sample
from ada_eval.paths import TEST_DATA_DIR


def setup_git_repo(repo_path: Path, *, initial_commit: bool = False):
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    if initial_commit:
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True
        )


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


def assert_log(caplog: pytest.LogCaptureFixture, level: int, message: str):
    """
    Assert that a message was logged.

    Args:
        caplog: The `LogCaptureFixture` to check.
        level: The log level (e.g., `logging.INFO`).
        message: The log message.

    Returns:
        The first matching log record.

    Raises:
        ValueError: If no matching log record is found.

    """
    for record in caplog.records:
        if record.levelno == level and record.message == message:
            return record
    raise ValueError(f"'{logging.getLevelName(level)}' message not found: {message}")


def dictify_datasets[SampleType: Sample](
    datasets: Sequence[Dataset[SampleType]],
) -> dict[tuple[str, str], SampleType]:
    """Construct a dictionary of samples keyed by (dataset.dirname, sample.name)."""
    return {(d.dirname, s.name): s for d in datasets for s in d.samples}


def _create_test_data_fixture(rel_path: Path) -> Callable[[Path], Path]:
    """
    Create a pytest fixture that copies test data into `tmp_path`.

    Args:
        rel_path: The relative path within `tests/data` to copy.

    Returns:
        A pytest fixture function.

    """
    assert not rel_path.is_absolute()

    @pytest.fixture
    def _fixture(tmp_path: Path) -> Path:
        dest = tmp_path / rel_path
        shutil.copytree(TEST_DATA_DIR / rel_path, dest)
        return dest

    return _fixture


expanded_test_datasets = _create_test_data_fixture(Path("valid_base_datasets/expanded"))
compacted_test_datasets = _create_test_data_fixture(
    Path("valid_base_datasets/compacted")
)
generated_test_datasets = _create_test_data_fixture(
    Path("valid_generated_datasets/compacted")
)
evaluated_test_datasets = _create_test_data_fixture(
    Path("valid_evaluated_datasets/compacted")
)
eval_test_datasets = _create_test_data_fixture(Path("eval_test_datasets"))
