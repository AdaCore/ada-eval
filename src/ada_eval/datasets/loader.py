from pathlib import Path
from typing import Any

from ada_eval.datasets.types import (
    CORRECT_STATEMENTS_KEY,
    INCORRECT_STATEMENTS_KEY,
    REFERENCE_ANSWER_FILE_NAME,
    AdaDataset,
    AdaSample,
    Dataset,
    DatasetType,
    ExplainDataset,
    ExplainSample,
    ExplainSolution,
    SparkDataset,
    SparkSample,
)
from ada_eval.datasets.types.datasets import (
    is_packed_dataset,
    is_unpacked_dataset,
)
from ada_eval.datasets.types.samples import is_unpacked_sample


class InvalidDatasetError(Exception):
    """Raised when a path is not a valid dataset."""

    def __init__(self, path: Path, dataset_type: str):
        super().__init__(f"{path} is not a {dataset_type} dataset")


class InvalidDatasetNameError(Exception):
    """Raised when a dataset name format is invalid."""

    def __init__(self, path: Path, expected_format: str):
        super().__init__(f"Expected {expected_format} to contain an underscore: {path}")


class UnknownDatasetTypeError(Exception):
    """Raised when an unknown dataset type is encountered."""

    def __init__(self, dataset_type):
        super().__init__(f"Unknown dataset type: {dataset_type}")


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


def load_unpacked_dataset(path: Path) -> Dataset:
    if not is_unpacked_dataset(path):
        raise InvalidDatasetError(path, "unpacked")
    if "_" not in path.stem:
        raise InvalidDatasetNameError(path, "unpacked dataset dir name")
    first_underscore = path.stem.index("_")
    dataset_type = DatasetType(path.stem[:first_underscore])
    match dataset_type:
        case DatasetType.ADA:
            sample_class = AdaSample
        case DatasetType.SPARK:
            sample_class = SparkSample
        case DatasetType.EXPLAIN:
            sample_class = ExplainSample
        case _:
            raise UnknownDatasetTypeError(dataset_type)
    dataset_name = path.stem[first_underscore + 1 :]
    samples = []
    for sample_dir in sorted(path.iterdir()):
        if not is_unpacked_sample(sample_dir):
            continue
        samples.append(sample_class.load_unpacked_sample(sample_dir))
    match dataset_type:
        case DatasetType.ADA:
            return AdaDataset(name=dataset_name, samples=samples, type=dataset_type)
        case DatasetType.EXPLAIN:
            return ExplainDataset(name=dataset_name, samples=samples, type=dataset_type)
        case DatasetType.SPARK:
            return SparkDataset(name=dataset_name, samples=samples, type=dataset_type)
        case _:
            raise UnknownDatasetTypeError(dataset_type)


def load_packed_dataset(path: Path) -> Dataset:
    if not is_packed_dataset(path):
        raise InvalidDatasetError(path, "packed")
    if "_" not in path.stem:
        raise InvalidDatasetNameError(path, "packed dataset filename")
    first_underscore = path.stem.index("_")
    dataset_type = DatasetType(path.stem[:first_underscore])
    dataset_name = path.stem[first_underscore + 1 :]
    match dataset_type:
        case DatasetType.ADA:
            dataset_class = AdaDataset
            sample_class = AdaSample
        case DatasetType.EXPLAIN:
            dataset_class = ExplainDataset
            sample_class = ExplainSample
        case DatasetType.SPARK:
            dataset_class = SparkDataset
            sample_class = SparkSample
        case _:
            raise UnknownDatasetTypeError(dataset_type)
    with path.open() as f:
        lines = f.readlines()
    samples = [sample_class.model_validate_json(x, strict=True) for x in lines]
    return dataset_class(name=dataset_name, samples=samples, type=dataset_type)
