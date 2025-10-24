import logging
from collections.abc import Sequence
from pathlib import Path

from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.trivial_generations import generate_canonical
from ada_eval.datasets.types import (
    Dataset,
    Eval,
    EvaluatedSample,
    GeneratedSample,
    Sample,
    dataset_has_sample_type,
    save_datasets_auto_format,
)
from ada_eval.evals.test import Test

from .build import Build
from .prove import Prove

logger = logging.getLogger(__name__)


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> Build | Prove | Test:
    match evaluation:
        case Eval.BUILD:
            return Build()
        case Eval.PROVE:
            return Prove()
        case Eval.TEST:
            return Test()
        case _:
            raise UnsupportedEvalError(evaluation)


def evaluate_datasets(
    evals: Sequence[Eval], datasets: Sequence[Dataset[GeneratedSample]], jobs: int
) -> list[Dataset[EvaluatedSample]]:
    """
    Run a list of `Eval`s on a list of `Dataset`s.

    All input datasets which are compatible with at least one of the evals will
    be included in the returned list.

    Args:
        evals: List of `Eval`s to run.
        datasets: List of datasets to evaluate.
        jobs: Number of parallel jobs to run.

    """
    if len(evals) == 0:
        logger.warning("No evals provided; skipping evaluation.")
        return []
    evaluations = [create_eval(e) for e in evals]
    for evaluation in evaluations:
        # `GenericEval` should catch exceptions raised during evaluation and
        # convert them to `EvaluationStatsFailed`, so any other exceptions
        # are unexpected and should propagate.
        evaluated_datasets, _, incompatible_datasets = evaluation.apply_to_datasets(
            datasets, jobs=jobs, catch_exceptions=False
        )
        if len(evaluated_datasets) > 0:
            # Recombine all datasets for the next eval (those incompatible with
            # this eval may be compatible with others).
            datasets = evaluated_datasets + incompatible_datasets
    return [
        dataset
        for dataset in datasets
        if dataset_has_sample_type(dataset, EvaluatedSample)
    ]


def evaluate_datasets_canonical[SampleType: Sample](
    evals: Sequence[Eval], datasets: Sequence[Dataset[SampleType]], jobs: int
) -> None:
    """
    Run a list of `Eval`s on the canonical solutions from a list of `Dataset`s.

    All input datasets will be included in the returned list, with the
    `canonical_evaluation_results` field of each sample merged with the new
    results from any compatible evaluations.

    The datsets and samples are modified in-place.

    Args:
        evals: List of `Eval`s to run.
        datasets: List of datasets to evaluate.
        jobs: Number of parallel jobs to run.

    """
    # Create a mapping from `(dataset.dirname, sample.name)` to the original
    # sample for future reference.
    original_samples = {(d.dirname, s.name): s for d in datasets for s in d.samples}
    # Create fake `GeneratedSample`s with the canonical solution as their
    # "generated" solutions, and evaluate these "generated" datasets.
    generated_datasets = generate_canonical(datasets)
    evaluated_datasets = evaluate_datasets(evals, generated_datasets, jobs=jobs)
    # Record the evaluation results in the `canonical_evaluation_results` field
    # of the original `Sample`s.
    for evaluated_dataset in evaluated_datasets:
        for evaluated_sample in evaluated_dataset.samples:
            original_sample = original_samples[
                (evaluated_dataset.dirname, evaluated_sample.name)
            ]
            # Merge new results with existing, overwriting only when we have
            # re-run the same eval.
            original_results = {
                es.eval: es for es in original_sample.canonical_evaluation_results
            }
            new_results = {es.eval: es for es in evaluated_sample.evaluation_results}
            combined_results = original_results | new_results
            original_sample.canonical_evaluation_results = list(
                combined_results.values()
            )


def evaluate_directory(
    evals: Sequence[Eval],
    path: Path,
    output_dir: Path,
    jobs: int,
    *,
    canonical_evaluation: bool = False,
) -> None:
    """
    Evaluate all samples in a file/directory and write the results to another.

    Args:
        evals: List of `Eval`s to run.
        path: Path to a packed dataset file, or a directory containing packed or
            unpacked dataset(s).
        output_dir: Directory where the results will be saved.
        jobs: Number of parallel jobs to run.
        canonical_evaluation: If `True`, evaluate the canonical solution instead
            of the generated solution. The datasets will not be promoted to
            `EvaluatedSample`s, and the results will instead be recorded in the
            `canonical_evaluation_results` field (merged with any results
            already present).

    """
    # Load datasets
    datasets_unchecked = load_datasets(path)
    # Evaluate datasets
    evaluated_datasets: Sequence[Dataset[Sample]]
    if canonical_evaluation:
        evaluate_datasets_canonical(evals, datasets_unchecked, jobs=jobs)
        evaluated_datasets = datasets_unchecked
    else:
        # Warn about datasets without generated solutions
        datasets: list[Dataset[GeneratedSample]] = []
        for dataset in datasets_unchecked:
            if dataset_has_sample_type(dataset, GeneratedSample):
                datasets.append(dataset)
            else:
                logger.warning(
                    "Dataset '%s' does not contain generations; Skipping evaluation.",
                    dataset.dirname,
                )
        # Evaluate all datasets
        evaluated_datasets = evaluate_datasets(evals, datasets, jobs=jobs)
        # Warn if nothing was evaluated
        if len(evaluated_datasets) == 0:
            logger.warning(
                "No datasets were compatible with any eval; no results to save."
            )
    # Save results to `output_dir` (respecting the format of any existing data)
    save_datasets_auto_format(evaluated_datasets, output_dir)
