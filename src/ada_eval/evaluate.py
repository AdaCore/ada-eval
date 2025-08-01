import dataclasses
import logging
from pathlib import Path

from ada_eval.datasets import Dataset, EvaluatedSample, Sample
from ada_eval.datasets.loader import load_dir
from ada_eval.evals import Eval, create_eval

logger = logging.getLogger(__name__)


def run_evals(
    evals: list[Eval],
    packed_dataset_or_dir: Path,
    output_dir: Path,
    jobs: int = 1,
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
    datasets = load_dir(packed_dataset_or_dir)
    # Evaluate all datasets
    for evaluation in evaluations:
        evaluated_datasets, incompatible_datasets = evaluation.apply_to_datasets(
            datasets,
            desc=f"Evaluating with {evaluation.name}",
            jobs=jobs,
        )
        if len(evaluated_datasets) == 0:
            logger.warning(
                "%s found nothing to evaluate at %s",
                evaluation.name,
                packed_dataset_or_dir,
            )
        else:
            # Combine evaluated datasets with incompatible datasets, relaxing
            # their type back to `Dataset[Sample]`
            datasets = [
                Dataset[Sample](**dataclasses.asdict(dataset))
                for dataset in evaluated_datasets
            ] + incompatible_datasets
    # Save results to `output_dir`
    output_dir.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        # Omit datasets with no evaluation results
        if issubclass(dataset.type, EvaluatedSample):
            dataset.save_packed(output_dir)
