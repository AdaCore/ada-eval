import shutil
from pathlib import Path

from ada_eval.datasets.types.datasets import Dataset, get_packed_dataset_files
from ada_eval.datasets.types.samples import Sample
from ada_eval.paths import EVALUATION_WORKING_DIR


def get_dataset_working_dir(dataset: Dataset) -> Path:
    return EVALUATION_WORKING_DIR / f"{dataset.type}_{dataset.name}"


def get_sample_working_dir(sample: Sample, dataset_working_dir: Path) -> Path:
    return dataset_working_dir / sample.name


def unpack_dataset_for_evaluation(dataset: Dataset):
    # Remove files from previous runs
    dataset_working_dir = get_dataset_working_dir(dataset)
    shutil.rmtree(dataset_working_dir, ignore_errors=True)

    # Don't ignore errors so we know if it fails to clean up previous files
    dataset_working_dir.mkdir()

    # Unpack each sample
    for sample in dataset.samples:
        sample_working_dir = get_sample_working_dir(sample, dataset_working_dir)
        sample_working_dir.mkdir()
        sample.unpack_for_generation(sample_working_dir)


def evaluate_completions(packed_dataset_or_dir: Path, jobs: int, output_dir: Path):  # noqa: ARG001
    dataset_files = get_packed_dataset_files(packed_dataset_or_dir)

    if len(dataset_files) == 0:
        print(f"No datasets could be found at: {packed_dataset_or_dir}")
        return

    EVALUATION_WORKING_DIR.mkdir(exist_ok=True)
    print("TODO evaluate: ", dataset_files)
    # TODO load the dataset (see generate.py for inspiration)
    # TODO Unpack all of the samples
    # TODO evaluate each of the samples
    # TODO write evaluation results to file
