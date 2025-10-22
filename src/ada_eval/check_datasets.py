import logging
from collections.abc import Sequence
from pathlib import Path

from ada_eval.datasets import Dataset, Sample, load_datasets
from ada_eval.datasets.types.datasets import verify_datasets_equal

logger = logging.getLogger(__name__)


class CanonicalEvaluationFailedError(ValueError):
    """Raised when a canonical evaluation result does not represent a pass."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        failed = [
            es.eval.value for es in sample.canonical_evaluation_results if not es.passed
        ]
        super().__init__(
            f"Sample '{sample.name}' of dataset '{dataset.dirname}' has "
            f"non-passing canonical evaluation results: {failed}"
        )


def check_canonical_evaluation_results(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Verify that all canonical evaluation results in `datasets` passed.

    Does not check that the results are accurate.

    Raises:
        CanonicalEvaluationFailedError: If any canonical evaluation result does
            not represent a passing result.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            for eval_stats in sample.canonical_evaluation_results:
                if not eval_stats.passed:
                    raise CanonicalEvaluationFailedError(dataset, sample)


def check_base_datasets(expanded_dir: Path, compacted_dir: Path) -> None:
    """
    Check the correctness of a set of base datasets.

    Checks that the datasets in `expanded_dir` and `compacted_dir` are
    equivalent.

    Raises:
        DatasetsMismatchError: If any difference is found between the expanded and
            and compacted datasets.
        CanonicalEvaluationFailedError: If any canonical evaluation result does
            not represent a passing result.

    """
    # Verify that the compacted and expanded datasets match
    expanded = load_datasets(expanded_dir)
    compacted = load_datasets(compacted_dir)
    verify_datasets_equal(
        expanded, "the expanded datasets", compacted, "the compacted datasets"
    )
    datasets = expanded
    # Verify that all canonical evaluation results represent passing results
    check_canonical_evaluation_results(datasets)
    # Report success
    logger.info("Base datasets are correct.")
