import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types import (
    GENERATED_SAMPLE_TYPES,
    Dataset,
    Eval,
    EvaluatedSample,
    GeneratedSample,
    GenerationStats,
    Sample,
    dataset_has_sample_type,
    save_datasets_auto_format,
)
from ada_eval.evals import create_eval

logger = logging.getLogger(__name__)


def evaluate_datasets(
    evals: list[Eval], datasets: list[Dataset[GeneratedSample]], jobs: int
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


SampleType = TypeVar("SampleType", bound=Sample)


def evaluate_datasets_canonical(
    evals: list[Eval], datasets: list[Dataset[SampleType]], jobs: int
) -> list[Dataset[SampleType]]:
    """
    Run a list of `Eval`s on the canonical solutions from a list of `Dataset`s.

    All input datasets will be included in the returned list, with the
    `canonical_evaluation_results` field of each sample merged with the new
    results from any compatible evaluations.

    The samples are modified in-place.

    Args:
        evals: List of `Eval`s to run.
        datasets: List of datasets to evaluate.
        jobs: Number of parallel jobs to run.

    """
    # Create a mapping from `(dataset_name, dataset_kind, sample_name)`
    # to the original sample for future reference.
    original_samples = {
        (dataset.name, dataset.kind, sample.name): sample
        for dataset in datasets
        for sample in dataset.samples
    }
    # Create fake `GeneratedSample`s with the canonical solution as their
    # "generated" solutions.
    dummy_gen_stats = GenerationStats(exit_code=0, stdout="", stderr="", runtime_ms=0)
    generated_datasets: list[Dataset[GeneratedSample]] = []
    for dataset in datasets:
        sample_type = GENERATED_SAMPLE_TYPES[dataset.kind]
        samples = [
            sample_type(
                **sample.model_dump(
                    exclude={
                        "generation_stats",
                        "generated_solution",
                        "evaluation_results",
                    }
                ),
                generation_stats=dummy_gen_stats,
                generated_solution=sample.canonical_solution,
            )
            for sample in dataset.samples
        ]
        generated_datasets.append(
            Dataset(name=dataset.name, sample_type=sample_type, samples=samples)
        )
    # Evaluate the "generated" datasets
    evaluated_datasets = evaluate_datasets(evals, generated_datasets, jobs=jobs)
    # Record the evaluation results in the `canonical_evaluation_results` field
    # of the original `Sample`s.
    for eval_dataset in evaluated_datasets:
        for evaluated_sample in eval_dataset.samples:
            original_sample = original_samples[
                (eval_dataset.name, eval_dataset.kind, evaluated_sample.name)
            ]
            # Merge new results with existing, overwriting only when we have
            # re-run the same eval.
            combined_results = {
                es.eval: es for es in original_sample.canonical_evaluation_results
            }
            combined_results.update(
                {es.eval: es for es in evaluated_sample.evaluation_results}
            )
            original_sample.canonical_evaluation_results = list(
                combined_results.values()
            )
    # Return the updated (original) `datasets`
    return datasets


def evaluate_directory(
    evals: list[Eval],
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
            `canonical_evaluation_results` field (overriding any value already
            present).

    """
    # Load datasets
    datasets_unchecked = load_datasets(path)
    # Evaluate datasets
    evaluated_datasets: Sequence[Dataset[Sample]]
    if canonical_evaluation:
        evaluated_datasets = evaluate_datasets_canonical(
            evals, datasets_unchecked, jobs=jobs
        )
    else:
        # Warn about datasets without generated solutions
        datasets: list[Dataset[GeneratedSample]] = []
        for dataset in datasets_unchecked:
            if dataset_has_sample_type(dataset, GeneratedSample):
                datasets.append(dataset)
            else:
                logger.warning(
                    "Dataset '%s' does not contain generations; Skipping evaluation.",
                    dataset.dirname(),
                )
        # Evaluate all datasets
        evaluated_datasets = evaluate_datasets(evals, datasets, jobs=jobs)
        # Warn if nothing was evaluated
        if len(evaluated_datasets) == 0:
            logger.warning(
                "No datasets were compatible with any eval; no results to save."
            )
            return
    # Save results to `output_dir` (respecting the format of any existing data)
    save_datasets_auto_format(evaluated_datasets, output_dir)
