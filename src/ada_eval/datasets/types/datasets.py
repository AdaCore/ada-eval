from __future__ import annotations

import logging
import shutil
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from ada_eval.utils import diff_dicts

from .samples import (
    GeneratedSample,
    Sample,
    SampleKind,
    SampleStage,
    is_unpacked_sample,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Dataset[SampleType: Sample]:
    """
    A dataset of samples.

    Attributes:
        name: Name of the dataset. Should be unique within the dataset kind.
        sample_type: The concrete type of all samples in the dataset.
        samples: The dataset's samples.

    """

    name: str
    sample_type: type[SampleType]
    samples: Sequence[SampleType]  # Must be immutable for covariance

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
        """The kind of this dataset."""
        return self.sample_type.kind

    @property
    def stage(self) -> SampleStage:
        """The stage of this dataset."""
        return self.sample_type.stage

    @property
    def dirname(self) -> str:
        """The stem of this dataset's file or directory name."""
        return f"{self.kind}_{self.name}"

    def save_unpacked(self, unpacked_datasets_root: Path):
        dataset_root = unpacked_datasets_root / self.dirname
        dataset_root.mkdir(exist_ok=True, parents=True)
        for sample in self.samples:
            sample.unpack(dataset_root)

    def save_packed(self, dest_dir: Path):
        dest_file = dest_dir / f"{self.dirname}.jsonl"
        with dest_file.open("w") as f:
            for sample in self.samples:
                f.write(sample.model_dump_json(exclude_defaults=True) + "\n")


def dataset_has_sample_type[TargetSampleType: Sample](
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
    if output_dir.is_file():
        output_dir.unlink()
    elif output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for dataset in datasets:
        (dataset.save_unpacked if unpacked else dataset.save_packed)(output_dir)


def save_datasets_auto_format(datasets: Sequence[Dataset[Sample]], path: Path) -> None:
    """
    Save datasets to a directory, respecting the format of any existing data thereat.

    The data will be saved in unpacked form if `path` already exists and
    contains unpacked data.

    If the output path points to a single dataset file/directory (instead of a
    directory thereof), and `datasets` contains exactly one dataset with a
    matching `dirname`, it will be saved as a single dataset file/directory in
    the same way.

    Any existing files at or within `path` will be removed or overwritten. A
    directory will be created if necessary (even if `datasets` is empty).

    Args:
        datasets: Datasets to save.
        path: File/Directory where the datasets will be saved.

    """
    unpacked = is_unpacked_data(path)
    if unpacked and is_packed_data(path):
        logger.warning(
            "Output path '%s' contains a mixture of packed and unpacked data; "
            "Defaulting to packed format.",
            path,
        )
        unpacked = False
    if is_unpacked_dataset(path) and is_collection_of_unpacked_datasets(path):
        logger.warning(
            "Output path '%s' contains a mixture of datasets and samples.", path
        )
    # Save as a single dataset if appropriate
    if len(datasets) == 1:
        dataset = datasets[0]
        if (
            unpacked
            and is_unpacked_dataset(path)
            and not is_collection_of_unpacked_datasets(path)
            and path.name == dataset.dirname
        ):
            shutil.rmtree(path)
            dataset.save_unpacked(path.parent)
            return
        if is_packed_dataset(path) and path.name == f"{dataset.dirname}.jsonl":
            dataset.save_packed(path.parent)
            return
    # Otherwise, save as a collection of datasets
    save_datasets(datasets, path, unpacked=unpacked)


class DatasetsMismatchError(ValueError):
    """Raised when two collections of datasets do not match."""


def verify_datasets_equal(
    datasets1: Collection[Dataset[Sample]],
    datasets1_name: str,
    datasets2: Collection[Dataset[Sample]],
    datasets2_name: str,
) -> None:
    """
    Verify that two collections of datasets are the same.

    The order of the datasets in each collection and the samples in each dataset
    does not matter.

    Args:
        datasets1: The first collection of datasets.
        datasets1_name: Name of the first collection (for exception messages).
        datasets2: The second collection of datasets.
        datasets2_name: Name of the second collection (for exception messages).

    Raises:
        DatasetsMismatchError: If the datasets do not match.

    """
    # Check for missing datasets by dirname
    datasets1_dict = {d.dirname: d for d in datasets1}
    datasets2_dict = {d.dirname: d for d in datasets2}
    for dirname in datasets1_dict.keys() ^ datasets2_dict.keys():
        present_in = datasets1_name if dirname in datasets1_dict else datasets2_name
        msg = f"dataset '{dirname}' is only present in {present_in}."
        raise DatasetsMismatchError(msg)
    # Check for any remaining differences in dataset type (this will only arise
    # from differences in sample stage, since the kind is part of the dirname)
    for dataset_dirname in datasets1_dict:  # noqa: PLC0206 (for symmetry)
        dataset1_type = datasets1_dict[dataset_dirname].sample_type
        dataset2_type = datasets2_dict[dataset_dirname].sample_type
        if dataset1_type is not dataset2_type:
            msg = (
                f"dataset '{dataset_dirname}' has type "
                f"'{dataset1_type.__name__}' in {datasets1_name} but type "
                f"'{dataset2_type.__name__}' in {datasets2_name}."
            )
            raise DatasetsMismatchError(msg)
        # Check for differences in the samples
        samples1 = {s.name: s for s in datasets1_dict[dataset_dirname].samples}
        samples2 = {s.name: s for s in datasets2_dict[dataset_dirname].samples}
        # Check for missing samples
        for sample_name in samples1.keys() ^ samples2.keys():
            present_in = datasets1_name if sample_name in samples1 else datasets2_name
            msg = (
                f"sample '{sample_name}' in dataset '{dataset_dirname}' is "
                f"only present in {present_in}."
            )
            raise DatasetsMismatchError(msg)
        # Check for differing samples
        for sample_name, sample1 in samples1.items():
            sample2 = samples2[sample_name]
            if sample1 != sample2:
                diff1, diff2 = diff_dicts(sample1.model_dump(), sample2.model_dump())
                msg = (
                    f"sample '{sample_name}' in dataset '{dataset_dirname}' "
                    f"differs between {datasets1_name} and {datasets2_name}:\n\n"
                    f"{diff1!r}\n\n{diff2!r}"
                )
                raise DatasetsMismatchError(msg)
