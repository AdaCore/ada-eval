import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from .types import (
    AdaSample,
    Dataset,
    DatasetKind,
    EvaluatedAdaSample,
    EvaluatedExplainSample,
    EvaluatedSample,
    EvaluatedSparkSample,
    ExplainSample,
    GeneratedAdaSample,
    GeneratedExplainSample,
    GeneratedSample,
    GeneratedSparkSample,
    Sample,
    SparkSample,
    get_packed_dataset_files,
    get_unpacked_dataset_dirs,
    is_packed_data,
    is_unpacked_data,
)
from .types.datasets import is_packed_dataset, is_unpacked_dataset
from .types.samples import is_unpacked_sample

logger = logging.getLogger(__name__)


class InvalidDatasetError(Exception):
    """Raised when a path is not a valid dataset."""

    def __init__(self, path: Path, dataset_type: str):
        super().__init__(f"'{path}' is not a valid {dataset_type} dataset")


class MixedDatasetFormatsError(ValueError):
    """Raised when loading a path containing both packed and unpacked datasets."""

    def __init__(self, path: Path):
        super().__init__(
            f"'{path}' contains a mixture of packed and unpacked datasets."
        )


class InvalidDatasetNameError(Exception):
    """Raised when a dataset name format is invalid."""

    def __init__(self, path: Path, expected_format: str):
        super().__init__(f"Expected {expected_format} to contain an underscore: {path}")


class UnknownDatasetKindError(Exception):
    """Raised when an unknown dataset type is encountered."""

    def __init__(self, dataset_type: DatasetKind | str):
        super().__init__(f"Unknown dataset type: {dataset_type}")


class DuplicateSampleNameError(ValueError):
    """Raised when a dataset contains more than one sample with the same name."""

    def __init__(self, sample_name: str, location: Path):
        super().__init__(f"Duplicate sample name '{sample_name}' found in '{location}'")


class MixedSampleTypesError(ValueError):
    """Raised when a dataset contains samples of different types."""

    def __init__(self, location: Path, samples: Sequence[Sample]):
        s1 = samples[0]
        s2 = next(s for s in samples if type(s) is not type(s1))
        super().__init__(
            f"Dataset at '{location}' contains mixed sample types:\n"
            f"'{s1.name}' is {type(s1).__name__} but '{s2.name}' is {type(s2).__name__}"
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
        DuplicateSampleNameError: If duplicate sample names are found.

    """
    seen_names = set()
    for sample in samples:
        if sample.name in seen_names:
            raise DuplicateSampleNameError(sample.name, location)
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


def _load_packed_sample(
    dataset_type: DatasetKind, line: tuple[int, str], path: Path
) -> Sample:
    """
    Load a single sample from a line of JSON in a packed dataset file.

    Tries to parse the line as an `EvaluatedSample`, `GeneratedSample` or
    `Sample` (in that order), raising a `ValidationError` if none succeed.

    Args:
        dataset_type: The type of dataset being loaded.
        line: A tuple containing the line number (1-indexed) and the line of
            JSON to parse.
        path: The path to the dataset file (for error messages).

    """
    base_sample_class: type[Sample]
    generated_sample_class: type[GeneratedSample]
    evaluated_sample_class: type[EvaluatedSample]
    match dataset_type:
        case DatasetKind.ADA:
            base_sample_class = AdaSample
            generated_sample_class = GeneratedAdaSample
            evaluated_sample_class = EvaluatedAdaSample
        case DatasetKind.EXPLAIN:
            base_sample_class = ExplainSample
            generated_sample_class = GeneratedExplainSample
            evaluated_sample_class = EvaluatedExplainSample
        case DatasetKind.SPARK:
            base_sample_class = SparkSample
            generated_sample_class = GeneratedSparkSample
            evaluated_sample_class = EvaluatedSparkSample
        case _:
            raise UnknownDatasetKindError(dataset_type)
    # Load each sample as an `EvaluatedSample`, `GeneratedSample` or `Sample`
    # (in that order).
    parse_order = (evaluated_sample_class, generated_sample_class, base_sample_class)
    for i, sample_class in enumerate(parse_order):
        try:
            sample = sample_class.model_validate_json(line[1], strict=True)
        except Exception as e:
            if isinstance(e, ValidationError) and i < len(parse_order) - 1:
                # Failed to validate, but there are more models to try
                continue
            # No more models to try, or a non-validation exception, so re-raise
            # the exception with a note
            e.add_note(f"This error occurred while parsing line {line[0]} of '{path}'")
            raise
        else:
            return sample
    raise RuntimeError("Unreachable")  # Ruff doesn't know loop always returns/raises


def load_packed_dataset(path: Path) -> Dataset[Sample]:
    """Load a packed dataset from its `.jsonl` file."""
    if not is_packed_dataset(path):
        raise InvalidDatasetError(path, "packed")
    dataset_type, dataset_name = _parse_dataset_dirname(path, "packed dataset filename")
    with path.open() as f:
        lines = [s for s in f.readlines() if s.strip() != ""]  # Ignore blank lines
    if len(lines) == 0:
        # Empty dataset should warn and return with most permissive type
        logger.warning("Dataset at '%s' is empty.", path)
        evaluated_sample_classes: dict[DatasetKind, type[EvaluatedSample]] = {
            DatasetKind.ADA: EvaluatedAdaSample,
            DatasetKind.SPARK: EvaluatedSparkSample,
            DatasetKind.EXPLAIN: EvaluatedExplainSample,
        }
        return Dataset(
            name=dataset_name,
            samples=[],
            sample_type=evaluated_sample_classes[dataset_type],
        )
    samples = [
        _load_packed_sample(dataset_type, line, path)
        for line in enumerate(lines, start=1)
    ]
    # All samples should be of the same concrete type, and names should be unique
    sample_class = type(samples[0])
    if not all(type(s) is sample_class for s in samples):
        raise MixedSampleTypesError(path, samples)
    check_no_duplicate_sample_names(samples, path)
    return Dataset(name=dataset_name, samples=samples, sample_type=sample_class)


def load_datasets(path: Path) -> list[Dataset[Sample]]:
    """
    Load all datasets in a file/directory.

    Load either a packed dataset file, a directory of packed datasets, an
    unpacked dataset directory or a directory of unpacked datasets.

    Args:
        path: The path to load.

    Returns:
        A list of loaded datasets.

    Raises:
        MixedDatasetFormatsError: If `path` contains a mixture of packed and
            unpacked datasets.
        InvalidDatasetNameError: If a dataset file has an invalid name format.
        UnknownDatasetKindError: If a dataset kind is not recognized.
        DuplicateSampleNameError: If a sample name is duplicated within a dataset
            (should only be possible for packed datasets on most file systems).
        PathMustBeRelativeError: If a sample's `Location` is not relative.
        json.decoder.JSONDecodeError: If a sample contains invalid JSON.
        pydantic.ValidationError: If a sample is invalid in some other way.

    """
    # Determine if we are loading packed or unpacked datasets
    packed = is_packed_data(path)
    unpacked = is_unpacked_data(path)
    if packed and unpacked:
        raise MixedDatasetFormatsError(path)
    # Load datasets
    if unpacked:
        dataset_paths = get_unpacked_dataset_dirs(path)
        datasets = [load_unpacked_dataset(path) for path in dataset_paths]
    else:
        dataset_paths = get_packed_dataset_files(path)
        datasets = [load_packed_dataset(path) for path in dataset_paths]
    if len(dataset_paths) == 0:
        logger.warning("No datasets could be found at: %s", path)
    return datasets
