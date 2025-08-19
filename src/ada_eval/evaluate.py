import logging
from collections.abc import Sequence
from pathlib import Path

from ada_eval.datasets import (
    Dataset,
    EvaluatedSample,
    GeneratedSample,
    Sample,
    dataset_has_sample_type,
)
from ada_eval.datasets.loader import load_dir
from ada_eval.datasets.types import (
    BASE_TYPE_TO_GENERATED,
    GenerationStats,
    is_unpacked_data,
    save_to_dir,
)
from ada_eval.evals import Eval, create_eval

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
    evaluations = [create_eval(e) for e in evals]
    for evaluation in evaluations:
        evaluated_datasets, failed_datasets, incompatible_datasets = (
            evaluation.apply_to_datasets(datasets, jobs=jobs)
        )
        if len(failed_datasets) > 0:
            # `GenericEval` should catch exceptions and convert them to
            # `EvaluationStatsFailed`, so this should be unreachable.
            msg = f"Unhandled exception during evaluation with {evaluation.name}."
            raise RuntimeError(msg)
        if len(evaluated_datasets) > 0:
            # Recombine all datasets for the next eval (those incompatible with
            # this eval may be compatible with others).
            datasets = evaluated_datasets + incompatible_datasets
    return [
        dataset
        for dataset in datasets
        if dataset_has_sample_type(dataset, EvaluatedSample)
    ]


def evaluate_datasets_canonical(
    evals: list[Eval], datasets: list[Dataset[Sample]], jobs: int
) -> list[Dataset[Sample]]:
    """
    Run a list of `Eval`s on the canonical solutions from a list of `Dataset`s.

    All input datasets will be included in the returned list, with the
    `canonical_evaluation_results` field of each sample overwritten with the
    results of any compatible evaluations.

    Args:
        evals: List of `Eval`s to run.
        datasets: List of datasets to evaluate.
        jobs: Number of parallel jobs to run.

    """
    # Create a mapping from `(dataset_name, dataset_kind, sample_name)`
    # to the original sample for future reference.
    original_samples = {
        (dataset.name, dataset.kind(), sample.name): sample
        for dataset in datasets
        for sample in dataset.samples
    }
    # Create fake `GeneratedSample`s with the canonical solution as their
    # "generated" solutions.
    dummy_gen_stats = GenerationStats(exit_code=0, stdout="", stderr="", runtime_ms=0)
    generated_datasets: list[Dataset[GeneratedSample]] = []
    for dataset in datasets:
        if dataset_has_sample_type(dataset, GeneratedSample):
            gen_sample_type = dataset.sample_type
        else:
            gen_sample_type = BASE_TYPE_TO_GENERATED[dataset.sample_type]
        gen_samples: list[GeneratedSample] = []
        for sample in dataset.samples:
            gen_sample = gen_sample_type(
                **sample.model_dump(),
                generation_stats=dummy_gen_stats,
                generated_solution=sample.canonical_solution,
            )
            if isinstance(gen_sample, EvaluatedSample):
                # Clear any existing evaluation results (which will not be
                # canonical)
                gen_sample.evaluation_results = []
            gen_samples.append(gen_sample)
        generated_dataset = Dataset(
            name=dataset.name, sample_type=gen_sample_type, samples=gen_samples
        )
        generated_datasets.append(generated_dataset)
    # Evaluate the "generated" datasets
    evaluated_datasets = evaluate_datasets(evals, generated_datasets, jobs=jobs)
    # Record the evaluation results in the `canonical_evaluation_results` field
    # of the original `Sample`s.
    for dataset in evaluated_datasets:
        for evaluated_sample in dataset.samples:
            original_sample = original_samples.get(
                (dataset.name, dataset.kind(), evaluated_sample.name)
            )
            if original_sample is not None:
                original_sample.canonical_evaluation_results = (
                    evaluated_sample.evaluation_results
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
    datasets_unchecked = load_dir(path)
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
                    dataset.name,
                )
        # Evaluate all datasets
        evaluated_datasets = evaluate_datasets(evals, datasets, jobs=jobs)
        # Warn if nothing was evaluated
        if len(evaluated_datasets) == 0:
            logger.warning(
                "No datasets were compatible with any eval; no results to save."
            )
            return
    # Save results to `output_dir` (avoiding overwriting unpacked data with
    # packed data, e.g. if running canonical evaluation on the unpacked base dir)
    save_to_dir(evaluated_datasets, output_dir, unpacked=is_unpacked_data(output_dir))
