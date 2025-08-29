from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeGuard, TypeVar

from .samples import (
    GeneratedSample,
    Sample,
    SampleKind,
    SampleStage,
    is_unpacked_sample,
)

SampleType_co = TypeVar("SampleType_co", bound=Sample, covariant=True)
TargetSampleType = TypeVar("TargetSampleType", bound=Sample)


@dataclass(kw_only=True)
class Dataset(Generic[SampleType_co]):
    """
    A dataset of samples.

    Attributes:
        name: Name of the dataset. Should be unique within the dataset kind.
        sample_type: The concrete type of all samples in the dataset.
        samples: The dataset's samples.

    """

    name: str
    sample_type: type[SampleType_co]
    samples: Sequence[SampleType_co]  # Must be immutable for covariance

    def __hash__(self) -> int:
        """Make Dataset hashable based on name and type only."""
        return hash((self.name, self.sample_type))

    def __eq__(self, other: object) -> bool:
        """Compare datasets based on name and type only."""
        if not isinstance(other, Dataset):
            return False
        return self.name == other.name and self.sample_type is other.sample_type

    @property
    def kind(self) -> SampleKind:
        """Return the kind of this dataset."""
        return self.sample_type.kind

    @property
    def stage(self) -> SampleStage:
        """Return the stage of this dataset."""
        return self.sample_type.stage

    def dirname(self) -> str:
        """Return the stem of this dataset's file or directory name."""
        return f"{self.kind}_{self.name}"

    def save_unpacked(self, unpacked_datasets_root: Path):
        dataset_root = unpacked_datasets_root / self.dirname()
        dataset_root.mkdir(exist_ok=True, parents=True)
        for sample in self.samples:
            sample.unpack(dataset_root)

    def save_packed(self, dest_dir: Path):
        dest_file = dest_dir / f"{self.dirname()}.jsonl"
        with dest_file.open("w") as f:
            for sample in self.samples:
                f.write(sample.model_dump_json(exclude_defaults=True) + "\n")


def dataset_has_sample_type(
    dataset: Dataset[Sample],
    sample_types: type[TargetSampleType] | Sequence[type[TargetSampleType]],
) -> TypeGuard[Dataset[TargetSampleType]]:
    """
    Type guard for a dataset's sample type.

    `dataset_has_sample_type(dataset, (A, B, ...))` is equivalent to
    `dataset_has_sample_type(dataset, A) or dataset_has_sample_type(dataset, B) or ...`.
    """
    if isinstance(sample_types, Sequence):
        return issubclass(dataset.sample_type, tuple(sample_types))
    return issubclass(dataset.sample_type, sample_types)


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


def is_unpacked_data(path: Path) -> bool:
    """Check if `path` is either an unpacked dataset or a collection thereof."""
    return is_unpacked_dataset(path) or is_collection_of_unpacked_datasets(path)


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


def is_packed_data(path: Path) -> bool:
    """Check if `path` is either a packed dataset or a collection thereof."""
    return is_packed_dataset(path) or is_collection_of_packed_datasets(path)


def get_packed_dataset_files(path: Path) -> list[Path]:
    """Get the list of paths to the files in the dataset."""
    if is_packed_dataset(path):
        return [path]
    if is_collection_of_packed_datasets(path):
        return [p for p in path.iterdir() if is_packed_dataset(p)]
    return []


def save_datasets(
    datasets: Iterable[Dataset[Sample]], output_dir: Path, *, unpacked: bool = False
) -> None:
    """
    Save datasets to a directory.

    Any existing files will be removed or overwritten. A directory will be
    created if necessary (even if `datasets` is empty).

    Args:
        datasets: Datasets to save.
        output_dir: Directory where the datasets will be saved.
        unpacked: If `True`, save the datasets in unpacked form, otherwise save
            them in packed form.

    """
    if unpacked and any(dataset_has_sample_type(d, GeneratedSample) for d in datasets):
        raise NotImplementedError(
            "Saving generated datasets in unpacked form is not supported."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for dataset in datasets:
        (dataset.save_unpacked if unpacked else dataset.save_packed)(output_dir)
