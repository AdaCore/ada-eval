import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ada_eval.datasets.types import (
    CORRECT_STATEMENTS_KEY,
    INCORRECT_STATEMENTS_KEY,
    REFERENCE_ANSWER_FILE_NAME,
    AdaSample,
    Dataset,
    DatasetKind,
    ExplainSample,
    ExplainSolution,
    GeneratedAdaSample,
    GeneratedExplainSample,
    GeneratedSample,
    GeneratedSparkSample,
    SparkSample,
    get_packed_dataset_files,
)
from ada_eval.datasets.types.datasets import (
    is_packed_dataset,
    is_unpacked_dataset,
)
from ada_eval.datasets.types.samples import Sample, is_unpacked_sample

logger = logging.getLogger(__name__)


class InvalidDatasetError(Exception):
    """Raised when a path is not a valid dataset."""

    def __init__(self, path: Path, dataset_type: str):
        super().__init__(f"{path} is not a {dataset_type} dataset")


class InvalidDatasetNameError(Exception):
    """Raised when a dataset name format is invalid."""

    def __init__(self, path: Path, expected_format: str):
        super().__init__(f"Expected {expected_format} to contain an underscore: {path}")


class UnknownDatasetKindError(Exception):
    """Raised when an unknown dataset type is encountered."""

    def __init__(self, dataset_type):
        super().__init__(f"Unknown dataset type: {dataset_type}")


class DuplicateNameError(ValueError):
    """Raised when a sample or dataset name is inappropriately duplicated."""

    def __init__(self, dataset_or_sample: Dataset[Sample] | Sample, location: Path):
        self.dataset_or_sample = dataset_or_sample
        if isinstance(dataset_or_sample, Dataset):
            name = f"{dataset_or_sample.dirname()})"
            object_type = "dataset"
        else:
            name = f"{dataset_or_sample.name}"
            object_type = "sample"
        super().__init__(f"Duplicate {object_type} name '{name}' found in '{location}'")


def get_explain_solution(
    sample_root: Path, other_data: dict[str, Any]
) -> ExplainSolution:
    file = sample_root / REFERENCE_ANSWER_FILE_NAME
    reference_answer = file.read_text(encoding="utf-8")
    return ExplainSolution(
        reference_answer=reference_answer,
        correct_statements=other_data[CORRECT_STATEMENTS_KEY],
        incorrect_statements=other_data[INCORRECT_STATEMENTS_KEY],
    )


def check_no_duplicate_sample_names(samples: Sequence[Sample], location: Path) -> None:
    """
    Check that no two samples in the sequence have the same name.

    Raises:
        DuplicateNameError: If duplicate sample names are found.

    """
    seen_names = set()
    for sample in samples:
        if sample.name in seen_names:
            raise DuplicateNameError(sample, location)
        seen_names.add(sample.name)


def load_unpacked_dataset(path: Path) -> Dataset[Sample]:
    if not is_unpacked_dataset(path):
        raise InvalidDatasetError(path, "unpacked")
    if "_" not in path.stem:
        raise InvalidDatasetNameError(path, "unpacked dataset dir name")
    dataset_type_str, _, dataset_name = path.stem.partition("_")
    dataset_type = DatasetKind(dataset_type_str)
    sample_class: type[Sample]
    match dataset_type:
        case DatasetKind.ADA:
            sample_class = AdaSample
        case DatasetKind.SPARK:
            sample_class = SparkSample
        case DatasetKind.EXPLAIN:
            sample_class = ExplainSample
        case _:
            raise UnknownDatasetKindError(dataset_type)
    samples = []
    for sample_dir in sorted(path.iterdir()):
        if not is_unpacked_sample(sample_dir):
            continue
        samples.append(sample_class.load_unpacked_sample(sample_dir))
    check_no_duplicate_sample_names(samples, path)
    match dataset_type:
        case DatasetKind.ADA:
            return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)
        case DatasetKind.EXPLAIN:
            return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)
        case DatasetKind.SPARK:
            return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)
        case _:
            raise UnknownDatasetKindError(dataset_type)


def load_packed_dataset(path: Path) -> Dataset[Sample]:
    """Load a packed dataset from its `.jsonl` file."""
    if not is_packed_dataset(path):
        raise InvalidDatasetError(path, "packed")
    if "_" not in path.stem:
        raise InvalidDatasetNameError(path, "packed dataset filename")
    dataset_type_str, _, dataset_name = path.stem.partition("_")
    dataset_type = DatasetKind(dataset_type_str)
    sample_class: type[Sample]
    generated_sample_class: type[GeneratedSample]
    match dataset_type:
        case DatasetKind.ADA:
            sample_class = AdaSample
            generated_sample_class = GeneratedAdaSample
        case DatasetKind.EXPLAIN:
            sample_class = ExplainSample
            generated_sample_class = GeneratedExplainSample
        case DatasetKind.SPARK:
            sample_class = SparkSample
            generated_sample_class = GeneratedSparkSample
        case _:
            raise UnknownDatasetKindError(dataset_type)
    with path.open() as f:
        lines = f.readlines()
    # Try to load as `GeneratedSample`s first, but fall back to `Sample`s if that fails
    samples: list[Sample]
    try:
        samples = [
            generated_sample_class.model_validate_json(x, strict=True) for x in lines
        ]
        check_no_duplicate_sample_names(samples, path)
        return Dataset(
            name=dataset_name, samples=samples, sample_type=generated_sample_class
        )
    except ValidationError:
        samples = [sample_class.model_validate_json(x, strict=True) for x in lines]
        check_no_duplicate_sample_names(samples, path)
        return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)


def load_dir(packed_dataset_or_dir: Path) -> list[Dataset[Sample]]:
    """
    Load all datasets in a file/directory.

    Args:
        packed_dataset_or_dir: Path to a packed dataset file, or a directory
            containing packed datasets.

    Returns:
        A list of loaded datasets.

    Raises:
        InvalidDatasetError: If the path is not a valid dataset.
        InvalidDatasetNameError: If a dataset file has an invalid name format.
        DuplicateNameError: If a dataset kind-name pair is duplicated, or a
            sample name is duplicated within a dataset.
        ValueError: If a dataset kind is not recognized.

    """
    # Load datasets
    dataset_files = get_packed_dataset_files(packed_dataset_or_dir)
    if len(dataset_files) == 0:
        logger.warning("No datasets could be found at: %s", packed_dataset_or_dir)
    datasets = [load_packed_dataset(path) for path in dataset_files]
    # Enforce unique dataset names
    datasets_set: set[tuple[str, DatasetKind]] = set()
    for dataset in datasets:
        name_and_kind = (dataset.name, dataset.kind())
        if name_and_kind in datasets_set:
            raise DuplicateNameError(dataset, packed_dataset_or_dir)
        datasets_set.add(name_and_kind)
    return datasets
