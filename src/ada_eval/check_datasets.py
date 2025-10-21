import logging
from pathlib import Path

from ada_eval.datasets import load_datasets
from ada_eval.datasets.types.datasets import verify_datasets_equal

logger = logging.getLogger(__name__)


def check_base_datasets(expanded_dir: Path, compacted_dir: Path) -> None:
    """
    Check the correctness of a set of base datasets.

    Checks that the datasets in `expanded_dir` and `compacted_dir` are
    equivalent.

    Raises:
        DatasetsMismatchError: If any difference is found between the expanded and
            and compacted datasets.

    """
    # Verify that the compacted and expanded datasets match
    expanded = load_datasets(expanded_dir)
    compacted = load_datasets(compacted_dir)
    verify_datasets_equal(
        expanded, "the expanded datasets", compacted, "the compacted datasets"
    )
    # Report success
    logger.info("Base datasets are correct.")
