from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from ada_eval.datasets.types.samples import is_unpacked_sample
from ada_eval.utils import UnexpectedTypeError

from .samples import (
    AdaSample,
    ExplainSample,
    Sample,
    SparkSample,
)


# Enum that specifies the kind of dataset (in terms of the base sample type,
# making no distinction between base, generated, or evaluated samples).
class DatasetKind(Enum):
    ADA = "ada"
    EXPLAIN = "explain"
    SPARK = "spark"

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_type(cls, sample_type: type[Sample]) -> DatasetKind:
        """Get the DatasetType from a sample type."""
        if issubclass(sample_type, SparkSample):
            return DatasetKind.SPARK
        elif issubclass(sample_type, AdaSample):
            return DatasetKind.ADA
        elif issubclass(sample_type, ExplainSample):
            return DatasetKind.EXPLAIN
        else:
            raise UnexpectedTypeError(expected_type=Sample, actual_type=sample_type)


SampleType = TypeVar("SampleType", bound=Sample)


@dataclass(kw_only=True)
class Dataset(Generic[SampleType]):
    name: str
    type: type[SampleType]
    samples: Sequence[SampleType]

    def __hash__(self) -> int:
        """Make Dataset hashable based on name and type only."""
        return hash((self.name, self.type))

    def __eq__(self, other: object) -> bool:
        """Compare datasets based on name and type only."""
        if not isinstance(other, Dataset):
            return False
        return self.name == other.name and self.type is other.type

    def dirname(self) -> str:
        """Get the stem of this dataset's file or directory name."""
        return f"{DatasetKind.from_type(self.type)}_{self.name}"

    def save_unpacked(self, unpacked_datasets_root: Path):
        dataset_root = unpacked_datasets_root / self.dirname()
        dataset_root.mkdir(exist_ok=True, parents=True)
        for sample in self.samples:
            sample.unpack(dataset_root)

    def save_packed(self, dest_dir: Path):
        dest_file = dest_dir / f"{self.dirname()}.jsonl"
        with dest_file.open("w") as f:
            for sample in self.samples:
                f.write(sample.model_dump_json() + "\n")


def is_unpacked_dataset(path: Path) -> bool:
    """
    Check if this is a path to a directory that contains an unpacked dataset.

    For this to be true, at least one child dir must contain a sample.
    """
    return path.is_dir() and any(is_unpacked_sample(d) for d in path.iterdir())


def is_collection_of_unpacked_datasets(path: Path) -> bool:
    """
    Check if this is a path to a directory that contains multiple datasets.

    Note that not all directories in this directory need to contain a dataset.
    """
    return path.is_dir() and any(is_unpacked_dataset(d) for d in path.iterdir())


def get_unpacked_dataset_dirs(path: Path) -> list[Path]:
    """Get the list of paths that contain the unpacked contents of a dataset."""
    if not is_collection_of_unpacked_datasets(path) and not is_unpacked_dataset(path):
        return []
    if is_unpacked_dataset(path):
        return [path]
    return [x for x in path.iterdir() if is_unpacked_dataset(x)]


def is_packed_dataset(path: Path) -> bool:
    """
    Check if this is the path to a packed dataset.

    This is the case if it's a path to a jsonl file.
    """
    return path.is_file() and path.suffix == ".jsonl"


def is_collection_of_packed_datasets(path: Path) -> bool:
    """
    Check if this is the path to a collection of packed datasets.

    This is the case if the dir contains a jsonl file.
    """
    return path.is_dir() and any(is_packed_dataset(p) for p in path.iterdir())


def get_packed_dataset_files(path: Path) -> list[Path]:
    """Get the list of paths to the files in the dataset."""
    if is_packed_dataset(path):
        return [path]
    if is_collection_of_packed_datasets(path):
        return [p for p in path.iterdir() if is_packed_dataset(p)]
    return []
