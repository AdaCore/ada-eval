import logging
from collections.abc import Sequence
from pathlib import Path

from ada_eval.datasets import Dataset, Eval, Sample, load_datasets
from ada_eval.datasets.types.datasets import verify_datasets_equal
from ada_eval.evals.evaluate import evaluate_datasets_canonical
from ada_eval.utils import diff_sequences

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


class WrongCanonicalEvaluationResultsError(ValueError):
    """Raised when a sample's `canonical_evaluation_results` are not accurate."""

    def __init__(self, dataset: Dataset[Sample], original: Sample, reevaluated: Sample):
        self.original_sample = original
        self.reevaluated_sample = reevaluated
        original_evals = {es.eval.value for es in original.canonical_evaluation_results}
        reevaluated_evals = {
            es.eval.value for es in reevaluated.canonical_evaluation_results
        }
        if original_evals != reevaluated_evals:
            super().__init__(
                f"Sample '{original.name}' of dataset '{dataset.dirname}' does "
                f"not have the expected set of canonical evaluation results:\n"
                f"{sorted(original_evals)} != {sorted(reevaluated_evals)}"
            )
        else:
            diff_left, diff_right = diff_sequences(
                original.model_dump()["canonical_evaluation_results"],
                reevaluated.model_dump()["canonical_evaluation_results"],
            )
            super().__init__(
                f"Mismatch found on re-evaluating sample '{original.name}' "
                f"of dataset '{dataset.dirname}':\n\n{diff_left}\n\n{diff_right}"
            )


def check_canonical_evaluation_results_accuracy(
    datasets: Sequence[Dataset[Sample]], jobs: int
) -> None:
    """
    Verify all canonical evaluation results in `datasets` actually match the samples.

    Re-runs a full canonical evaluation to check.

    Raises:
        WrongCanonicalEvaluationResultsError: If any of the current canonical
            evaluation results do not match the re-evaluated results.

    """
    reeval_datasets = [
        Dataset(
            name=dataset.name,
            sample_type=dataset.sample_type,
            samples=[sample.model_copy(deep=True) for sample in dataset.samples],
        )
        for dataset in datasets
    ]
    evaluate_datasets_canonical(list(Eval), reeval_datasets, jobs=jobs)
    for dataset, reeval_dataset in zip(datasets, reeval_datasets, strict=True):
        original_samples = {sample.name: sample for sample in dataset.samples}
        for reeval_sample in reeval_dataset.samples:
            original_sample = original_samples[reeval_sample.name]
            if (
                original_sample.canonical_evaluation_results
                != reeval_sample.canonical_evaluation_results
            ):
                raise WrongCanonicalEvaluationResultsError(
                    dataset, original_sample, reeval_sample
                )


def check_base_datasets(expanded_dir: Path, compacted_dir: Path, jobs: int) -> None:
    """
    Check the correctness of a set of base datasets.

    Checks that:
    - the datasets in `expanded_dir` and `compacted_dir` are equivalent.
    - all canonical evaluation results in the datasets represent passing results.
    - all canonical evaluation results in the datasets are accurate.

    Raises:
        DatasetsMismatchError: If any difference is found between the expanded and
            and compacted datasets.
        CanonicalEvaluationFailedError: If any canonical evaluation result does
            not represent a passing result.
        WrongCanonicalEvaluationResultsError: If any of the current canonical
            evaluation results do not match the results of a re-run of a full
            canonical evaluation.

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
    # Re-run a full canonical evaluation and verify that we get the same results
    # as those stored already.
    logger.info("Re-evaluating to check canonical evaluation results are correct ...")
    check_canonical_evaluation_results_accuracy(datasets, jobs=jobs)
    # Report success
    logger.info("Base datasets are correct.")
