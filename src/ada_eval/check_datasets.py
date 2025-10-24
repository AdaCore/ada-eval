import logging
from collections.abc import Sequence
from pathlib import Path

from ada_eval.datasets import (
    AdaSample,
    Dataset,
    Eval,
    EvaluatedSample,
    EvaluationStatsFailed,
    EvaluationStatsTimedOut,
    Sample,
    load_datasets,
)
from ada_eval.datasets.trivial_generations import generate_canonical, generate_null
from ada_eval.datasets.types.datasets import verify_datasets_equal
from ada_eval.evals.evaluate import evaluate_datasets
from ada_eval.utils import diff_sequences, serialise_sequence

logger = logging.getLogger(__name__)


class CanonicalEvaluationFailedError(ValueError):
    """Raised when a canonical evaluation result does not represent a pass."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        failed = [
            es.eval.value for es in sample.canonical_evaluation_results if not es.passed
        ]
        super().__init__(
            f"sample '{sample.name}' of dataset '{dataset.dirname}' has "
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

    def __init__(self, dataset: Dataset[Sample], reevaluated_sample: EvaluatedSample):
        original_evals = {
            es.eval.value for es in reevaluated_sample.canonical_evaluation_results
        }
        reevaluated_evals = {
            es.eval.value for es in reevaluated_sample.evaluation_results
        }
        if original_evals != reevaluated_evals:
            super().__init__(
                f"sample '{reevaluated_sample.name}' of dataset '{dataset.dirname}' "
                f"does not have the expected set of canonical evaluation results:\n"
                f"{sorted(original_evals)} != {sorted(reevaluated_evals)}"
            )
        else:
            diff_left, diff_right = diff_sequences(
                serialise_sequence(reevaluated_sample.canonical_evaluation_results),
                serialise_sequence(reevaluated_sample.evaluation_results),
            )
            super().__init__(
                f"mismatch found on re-evaluating sample '{reevaluated_sample.name}' "
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
    generated_datasets = generate_canonical(datasets)
    reevaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in reevaluated_datasets:
        for sample in dataset.samples:
            sample.canonical_evaluation_results = sorted(
                sample.canonical_evaluation_results, key=lambda es: es.eval.value
            )
            sample.evaluation_results = sorted(
                sample.evaluation_results, key=lambda es: es.eval.value
            )
            if sample.canonical_evaluation_results != sample.evaluation_results:
                raise WrongCanonicalEvaluationResultsError(dataset, sample)


class BaselineEvaluationPassedError(ValueError):
    """Raised when a no-op generation passes all evals for a sample."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        desc = (
            "the unmodified sources of"
            if isinstance(sample, AdaSample)
            else "the empty string for"
        )
        super().__init__(
            f"all evaluations passed on {desc} sample '{sample.name}' of "
            f"dataset '{dataset.dirname}'."
        )


class EvaluationError(ValueError):
    """Raised when an error or timeout occurs during evaluation."""

    def __init__(self, dataset: Dataset[Sample], sample: EvaluatedSample):
        self.dataset, self.sample = dataset, sample
        bad_eval_stats = next(
            es
            for es in sample.evaluation_results
            if isinstance(es, (EvaluationStatsTimedOut, EvaluationStatsFailed))
        )
        super().__init__(
            f"error during baseline evaluation of sample '{sample.name}' of "
            f"dataset '{dataset.dirname}': {bad_eval_stats!r}"
        )


def check_evaluation_baseline(datasets: Sequence[Dataset[Sample]], jobs: int) -> None:
    """
    Check that a no-op generation fails at least one eval for all samples in `datasets`.

    Also check that these failures were not due to timeouts or errors in the
    evaluations themselves.

    Raises:
        BaselineEvaluationPassedError: If any sample passes all evals when
            evaluated with a no-op generation.
        EvaluationError: If any sample encounters an error or timeout during
            evaluation.

    """
    generated_datasets = generate_null(datasets)
    evaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            if all(es.passed for es in sample.evaluation_results):
                raise BaselineEvaluationPassedError(dataset, sample)
            if any(
                isinstance(es, (EvaluationStatsTimedOut, EvaluationStatsFailed))
                for es in sample.evaluation_results
            ):
                raise EvaluationError(dataset, sample)


def check_base_datasets(expanded_dir: Path, compacted_dir: Path, jobs: int) -> None:
    """
    Check the correctness of a set of base datasets.

    Checks that:
    - the datasets in `expanded_dir` and `compacted_dir` are equivalent.
    - all canonical evaluation results in the datasets represent passing results.
    - all canonical evaluation results in the datasets are accurate.
    - a no-op generation fails at least one eval on all samples.

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
    # Check that a no-op generation fails at least one eval on all samples
    logger.info("Checking evaluation baseline ...")
    check_evaluation_baseline(datasets, jobs=jobs)
    # Report success
    logger.info("Base datasets are correct.")
