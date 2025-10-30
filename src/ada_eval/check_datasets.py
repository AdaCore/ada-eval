import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from ada_eval.datasets import (
    Dataset,
    Eval,
    EvaluatedSample,
    EvaluationStats,
    EvaluationStatsInvalid,
    ExplainSample,
    Sample,
    load_datasets,
    verify_datasets_equal,
)
from ada_eval.datasets.trivial_generations import generate_canonical, generate_null
from ada_eval.evals.evaluate import evaluate_datasets
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR
from ada_eval.utils import diff_sequences, serialise_sequence

logger = logging.getLogger(__name__)


class IncorrectDatasetError(ValueError):
    """Parent class for all dataset check exceptions."""


class FailedCanonicalEvaluationError(IncorrectDatasetError):
    """Raised when a canonical evaluation result does not represent a pass."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        failed = [
            es.eval.value for es in sample.canonical_evaluation_results if not es.passed
        ]
        super().__init__(
            f"sample '{sample.name}' in dataset '{dataset.dirname}' has "
            f"non-passing canonical evaluation results: {failed}"
        )


def check_canonical_evaluation_results(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Verify that all canonical evaluation results in `datasets` passed.

    Does not check that the results are accurate.

    Raises:
        FailedCanonicalEvaluationError: If any canonical evaluation result does
            not represent a passing result.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            for eval_stats in sample.canonical_evaluation_results:
                if not eval_stats.passed:
                    raise FailedCanonicalEvaluationError(dataset, sample)


class InaccurateCanonicalEvaluationError(IncorrectDatasetError):
    """Raised when a sample's `canonical_evaluation_results` are not accurate."""

    def __init__(
        self,
        dataset: Dataset[Sample],
        sample: Sample,
        expected: Collection[EvaluationStats],
    ) -> None:
        self.dataset, self.sample, self.expected = dataset, sample, expected
        actual_evals = {es.eval.value for es in sample.canonical_evaluation_results}
        expected_evals = {es.eval.value for es in expected}
        if actual_evals != expected_evals:
            super().__init__(
                f"sample '{sample.name}' in dataset '{dataset.dirname}' does "
                f"not have the expected set of canonical evaluation results:\n"
                f"{sorted(actual_evals)} != {sorted(expected_evals)}"
            )
        else:
            actual = sorted(sample.canonical_evaluation_results, key=lambda es: es.eval)
            expected = sorted(expected, key=lambda es: es.eval)
            diff_left, diff_right = diff_sequences(
                serialise_sequence(actual), serialise_sequence(expected)
            )
            super().__init__(
                f"mismatch found on re-evaluating sample '{sample.name}' "
                f"in dataset '{dataset.dirname}':\n\n{diff_left}\n\n{diff_right}"
            )


def check_canonical_evaluation_results_accuracy(
    datasets: Sequence[Dataset[Sample]], jobs: int
) -> None:
    """
    Verify all canonical evaluation results in `datasets` actually match the samples.

    Re-runs a full canonical evaluation to check.

    Raises:
        InaccurateCanonicalEvaluationError: If any of the current canonical
            evaluation results do not match the re-evaluated results.

    """
    generated_datasets = generate_canonical(datasets)
    reevaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in reevaluated_datasets:
        for sample in dataset.samples:
            actual = sorted(sample.canonical_evaluation_results, key=lambda es: es.eval)
            expected = sorted(sample.evaluation_results, key=lambda es: es.eval)
            if actual != expected:
                raise InaccurateCanonicalEvaluationError(dataset, sample, expected)
    # `evaluate_datasets()` only returns datasets compatible with at least one
    # eval. Incompatible datasets should have empty results.
    compatible_datasets = {d.dirname for d in reevaluated_datasets}
    for initial_dataset in datasets:
        if initial_dataset.dirname not in compatible_datasets:
            for initial_sample in initial_dataset.samples:
                if len(initial_sample.canonical_evaluation_results) > 0:
                    raise InaccurateCanonicalEvaluationError(
                        initial_dataset, initial_sample, []
                    )


class PassedBaselineEvaluationError(IncorrectDatasetError):
    """Raised when a null generation passes all evals for a sample."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        desc = (
            "the empty string for"
            if isinstance(sample, ExplainSample)
            else "the unmodified sources of"
        )
        super().__init__(
            f"all evaluations passed on {desc} sample '{sample.name}' in "
            f"dataset '{dataset.dirname}'."
        )


class InvalidBaselineEvaluationError(IncorrectDatasetError):
    """Raised when an error or timeout occurs during evaluation."""

    def __init__(self, dataset: Dataset[Sample], sample: EvaluatedSample):
        self.dataset, self.sample = dataset, sample
        bad_eval_stats = next(
            es
            for es in sample.evaluation_results
            if isinstance(es, EvaluationStatsInvalid)
        )
        super().__init__(
            f"error during baseline evaluation of sample '{sample.name}' in "
            f"dataset '{dataset.dirname}': {bad_eval_stats!r}"
        )


def check_evaluation_baseline(datasets: Sequence[Dataset[Sample]], jobs: int) -> None:
    """
    Check that a null generation fails at least one eval for all samples in `datasets`.

    Also check that these failures were not due to timeouts or errors in the
    evaluations themselves.

    Raises:
        PassedBaselineEvaluationError: If any sample passes all evals when
            evaluated with a null generation.
        InvalidBaselineEvaluationError: If any sample encounters an error or
            timeout during evaluation of the null generation.

    """
    generated_datasets = generate_null(datasets)
    evaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            if all(es.passed for es in sample.evaluation_results):
                raise PassedBaselineEvaluationError(dataset, sample)
            if any(
                isinstance(es, EvaluationStatsInvalid)
                for es in sample.evaluation_results
            ):
                raise InvalidBaselineEvaluationError(dataset, sample)


def check_base_datasets(dataset_dirs: Sequence[Path], jobs: int) -> None:
    """
    Check the correctness of a collection of base datasets.

    Checks that:
    - the specified directories all contain identical datasets.
    - all canonical evaluation results in the datasets represent passing results.
    - all canonical evaluation results in the datasets are accurate.
    - a null generation fails at least one eval on all samples.

    Raises:
        DatasetsMismatchError: If any difference is found between any of the
            datasets.
        FailedCanonicalEvaluationError: If any canonical evaluation result does
            not represent a passing result.
        InaccurateCanonicalEvaluationError: If any of the current canonical
            evaluation results do not match the results of a re-run of a full
            canonical evaluation.
        PassedBaselineEvaluationError: If any sample passes all evals when
            evaluated with a null generation.
        InvalidBaselineEvaluationError: If any sample encounters an error or
            timeout during evaluation of the null generation.

    """
    if len(dataset_dirs) == 0:
        logger.info("No dataset directories specified; nothing to check.")
        return
    datasets = load_datasets(dataset_dirs[0])
    # Verify that all datasets are equivalent
    dataset_names = {
        EXPANDED_DATASETS_DIR: "the expanded datasets",
        COMPACTED_DATASETS_DIR: "the compacted datasets",
    }
    for other_datasets_dir in dataset_dirs[1:]:
        verify_datasets_equal(
            datasets,
            dataset_names.get(dataset_dirs[0].resolve(), f"'{dataset_dirs[0]}'"),
            load_datasets(other_datasets_dir),
            dataset_names.get(other_datasets_dir.resolve(), f"'{other_datasets_dir}'"),
        )
    # Verify that all canonical evaluation results represent passing results
    check_canonical_evaluation_results(datasets)
    # Re-run a full canonical evaluation and verify that we get the same results
    # as those stored already.
    logger.info("Re-evaluating to check canonical evaluation results are accurate ...")
    check_canonical_evaluation_results_accuracy(datasets, jobs=jobs)
    # Check that a null generation fails at least one eval on all samples
    logger.info("Checking evaluation baseline ...")
    check_evaluation_baseline(datasets, jobs=jobs)
    # Report success
    logger.info("Base datasets are correct.")
