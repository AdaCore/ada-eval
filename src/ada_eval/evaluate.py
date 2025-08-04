import logging
from pathlib import Path

from ada_eval.datasets import (
    Dataset,
    EvaluatedSample,
    GeneratedSample,
    dataset_has_sample_type,
)
from ada_eval.datasets.loader import load_dir
from ada_eval.evals import Eval, create_eval

logger = logging.getLogger(__name__)


def run_evals(
    evals: list[Eval],
    packed_dataset_or_dir: Path,
    output_dir: Path,
    jobs: int,
) -> None:
    """
    Evaluate all samples in a file/directory and write the results to another.

    Args:
        evals: List of `Eval`s to run.
        packed_dataset_or_dir: Path to a packed dataset file, or a directory
            containing packed datasets.
        output_dir: Directory where the results will be saved.
        jobs: Number of parallel jobs to run.

    """
    # Instantiate evals
    evaluations = [create_eval(e) for e in evals]
    # Load datasets
    datasets_unchecked = load_dir(packed_dataset_or_dir)
    # Check for datasets without generated solutions
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
    for evaluation in evaluations:
        evaluated_datasets, failed_datasets, incompatible_datasets = (
            evaluation.apply_to_datasets(datasets, jobs=jobs)
        )
        if len(failed_datasets) > 0:
            # `GenericEval` should catch exceptions and convert them to
            # `EvaluationStatsFailed`, so something is wrong.
            msg = f"Unhandled exception during evaluation with {evaluation.name}."
            raise RuntimeError(msg)
        if len(evaluated_datasets) > 0:
            # Recombine all datasets for the next eval (those incompatible with
            # this eval may be compatible with others).
            datasets = evaluated_datasets + incompatible_datasets
    # Save results to `output_dir`, omitting datasets with no evaluation results
    if any(dataset_has_sample_type(dataset, EvaluatedSample) for dataset in datasets):
        output_dir.mkdir(exist_ok=True, parents=True)
        for dataset in datasets:
            if dataset_has_sample_type(dataset, EvaluatedSample):
                dataset.save_packed(output_dir)
    else:
        logger.warning("No datasets were compatible with any eval; no results to save.")
