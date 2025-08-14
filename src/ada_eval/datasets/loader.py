import logging
from collections.abc import Iterable
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
    get_unpacked_dataset_dirs,
    is_packed_dataset,
    is_unpacked_dataset,
)
from ada_eval.datasets.types.samples import Sample, is_unpacked_sample

logger = logging.getLogger(__name__)


class InvalidDatasetError(Exception):
    """Raised when a path is not a valid dataset."""

    def __init__(self, path: Path, dataset_type: str):
        super().__init__(f"'{path}' is not a valid {dataset_type} dataset")


class InvalidDatasetNameError(Exception):
    """Raised when a dataset name format is invalid."""

    def __init__(self, path: Path, expected_format: str):
        super().__init__(f"Expected {expected_format} to contain an underscore: {path}")


class UnknownDatasetKindError(Exception):
    """Raised when an unknown dataset type is encountered."""

    def __init__(self, dataset_type: DatasetKind | str):
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


def _parse_dataset_dirname(path: Path, expected_format: str) -> tuple[DatasetKind, str]:
    """Parse a dataset file/directory path into its kind and name."""
    if "_" not in path.stem:
        raise InvalidDatasetNameError(path, expected_format)
    dataset_type_str, _, dataset_name = path.stem.partition("_")
    if not any(k.value == dataset_type_str for k in DatasetKind):
        raise UnknownDatasetKindError(dataset_type_str)
    return DatasetKind(dataset_type_str), dataset_name


def check_no_duplicate_sample_names(samples: Iterable[Sample], location: Path) -> None:
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
    dataset_type, dataset_name = _parse_dataset_dirname(
        path, "unpacked dataset dir name"
    )
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
            logger.warning("Skipping non-sample directory: %s", sample_dir)
            continue
        try:
            loaded_sample = sample_class.load_unpacked_sample(sample_dir)
        except Exception as e:
            e.add_note(
                f"This exception occurred while loading the sample at: {sample_dir}"
            )
            raise
        samples.append(loaded_sample)
    check_no_duplicate_sample_names(samples, path)
    return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)


def load_packed_dataset(path: Path) -> Dataset[Sample]:
    """Load a packed dataset from its `.jsonl` file."""
    if not is_packed_dataset(path):
        raise InvalidDatasetError(path, "packed")
    dataset_type, dataset_name = _parse_dataset_dirname(path, "packed dataset filename")
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
        sample_class = generated_sample_class
    except ValidationError:
        samples = []
        for line_num, line in enumerate(lines, start=1):
            try:
                sample = sample_class.model_validate_json(line, strict=True)
            except Exception as e:
                e.add_note(
                    f"This error occurred while parsing line {line_num} of '{path}'"
                )
                raise
            samples.append(sample)
    check_no_duplicate_sample_names(samples, path)
    return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)


def load_dir(path: Path, *, unpacked: bool = False) -> list[Dataset[Sample]]:
    """
    Load all datasets in a file/directory.

    If `unpacked` is `False`, load either a packed dataset file or a directory
    containing packed datasets. If `unpacked` is `True`, load either an unpacked
    dataset directory or a directory of unpacked datasets.

    Args:
        path: The path to load.
        unpacked: Whether the datasets are unpacked.

    Returns:
        A list of loaded datasets.

    Raises:
        InvalidDatasetNameError: If a dataset file has an invalid name format.
        DuplicateNameError: If a dataset kind-name pair is duplicated (should be
            impossible on most file systems), or a sample name is duplicated
            within a dataset (should only be possible for packed datasets on
            most file systems).
        UnknownDatasetKindError: If a dataset kind is not recognized.
        PathMustBeRelativeError: If a sample's `Location` is not relative.
        json.decoder.JSONDecodeError: If a sample contains invalid JSON.
        pydantic.ValidationError: If a sample is invalid in some other way.

    """
    # Load datasets
    if unpacked:
        dataset_paths = get_unpacked_dataset_dirs(path)
        datasets = [load_unpacked_dataset(path) for path in dataset_paths]
    else:
        dataset_paths = get_packed_dataset_files(path)
        datasets = [load_packed_dataset(path) for path in dataset_paths]
    if len(dataset_paths) == 0:
        logger.warning("No datasets could be found at: %s", path)
    # Enforce unique dataset names
    datasets_set: set[tuple[str, DatasetKind]] = set()
    for dataset in datasets:
        name_and_kind = (dataset.name, dataset.kind())
        if name_and_kind in datasets_set:
            raise DuplicateNameError(dataset, path)
        datasets_set.add(name_and_kind)
    return datasets
