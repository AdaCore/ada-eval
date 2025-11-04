import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from ada_eval.paths import TEST_DATA_DIR


def _test_data_fixture(rel_path: Path) -> Callable[[Path], Path]:
    """
    Create a pytest fixture that copies test data into `tmp_path`.

    Args:
        rel_path: The relative path within `tests/data/` to copy.

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


expanded_test_datasets = _test_data_fixture(Path("valid_base_datasets/expanded"))
compacted_test_datasets = _test_data_fixture(Path("valid_base_datasets/compacted"))
generated_test_datasets = _test_data_fixture(Path("valid_generated_datasets/compacted"))
evaluated_test_datasets = _test_data_fixture(Path("valid_evaluated_datasets/compacted"))
eval_test_datasets = _test_data_fixture(Path("eval_test_datasets"))
check_test_datasets = _test_data_fixture(Path("check_test_datasets"))
