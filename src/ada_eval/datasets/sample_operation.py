import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generic, TypeVar

from tqdm import tqdm

from .loader import load_dir
from .types import Dataset, Sample, dataset_has_sample_type

logger = logging.getLogger(__name__)


class UnsupportedSampleTypeError(TypeError):
    """Raised when a sample type is not supported by a `SampleOperation`."""

    def __init__(self, sample_type, expected_types: tuple[type[Sample], ...]) -> None:
        if len(expected_types) == 1:
            expected_types_str = expected_types[0].__name__
        else:
            expected_types_str = f"one of {[t.__name__ for t in expected_types]}"
        super().__init__(
            f"Unsupported sample type: got {sample_type.__name__}, "
            "expected " + expected_types_str
        )


InputType = TypeVar("InputType", bound=Sample)
OutputType = TypeVar("OutputType", bound=Sample)
DatasetSampleType = TypeVar("DatasetSampleType", bound=Sample)


class SampleOperation(ABC, Generic[InputType, OutputType]):
    """
    An operation that converts one type of `Sample` to another.

    Args:
        type_mapping: A mapping from input `Sample` types to the corresponding
            output `Sample` types they will be converted into. Types not present
            in this mapping are considered unsupported by the operation.

    """

    _type_mapping: dict[type[InputType], type[OutputType]]

    def __init__(self, type_mapping: dict[type[InputType], type[OutputType]]):
        self._type_mapping = type_mapping

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, sample: InputType) -> OutputType:
        """Apply the operation to a sample."""

    def apply_to_datasets(
        self, datasets: Iterable[Dataset[DatasetSampleType]], desc: str, jobs: int
    ) -> tuple[
        list[Dataset[OutputType]],
        list[Dataset[InputType]],
        list[Dataset[DatasetSampleType]],
    ]:
        """
        Apply the operation to all samples in a collection of datasets.

        Args:
            datasets: Iterable of datasets to apply the operation to.
            desc: Description for progress tracking.
            jobs: Number of parallel jobs to run.

        Returns:
            transformed_datasets: New datasets with samples transformed by the
                operation.
            failed_datasets: Samples for which the operation failed, grouped by
                their original dataset. These samples will be omitted from
                `transformed_datasets`.
            incompatible_datasets: Datasets that were of types unsupported by
                the operation, and were therefore skipped.

        """
        # Filter by input type compatibility
        compatible_datasets: list[Dataset[InputType]] = []
        incompatible_datasets: list[Dataset[DatasetSampleType]] = []
        for inp_dataset in datasets:
            if dataset_has_sample_type(inp_dataset, tuple(self._type_mapping.keys())):
                compatible_datasets.append(inp_dataset)
            else:
                incompatible_datasets.append(inp_dataset)
        if len(compatible_datasets) == 0:
            logger.warning("No datasets compatible with %s found.", self.name)
            return [], [], incompatible_datasets

        # Calculate total number of samples for progress tracking
        total_samples = sum(len(dataset.samples) for dataset in compatible_datasets)

        # Apply to each sample
        dataset_results: dict[Dataset[InputType], list[OutputType]] = {
            dataset: [] for dataset in compatible_datasets
        }
        failures: dict[Dataset[InputType], list[InputType]] = {
            dataset: [] for dataset in compatible_datasets
        }
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all futures and create a mapping from each future to its
            # input sample and the dataset it came from.
            future_to_input = {
                executor.submit(self.apply, sample): (dataset, sample)
                for dataset in compatible_datasets
                for sample in dataset.samples
            }

            # Process futures as they complete with progress tracking
            with tqdm(total=total_samples, desc=desc) as pbar:
                for future in as_completed(future_to_input.keys()):
                    dataset, sample = future_to_input[future]
                    try:
                        result = future.result()
                        dataset_results[dataset].append(result)
                    except Exception:
                        logging.exception("Error processing sample")
                        failures[dataset].append(sample)
                    finally:
                        pbar.update(1)

        # Create new datasets with transformed samples
        new_datasets: list[Dataset[OutputType]] = [
            Dataset(
                name=old_dataset.name,
                sample_type=self._type_mapping[old_dataset.sample_type],
                samples=results,
            )
            for old_dataset, results in dataset_results.items()
            if len(results) > 0
        ]
        failed_datasets: list[Dataset[InputType]] = [
            Dataset(
                name=old_dataset.name,
                sample_type=old_dataset.sample_type,
                samples=failed_samples,
            )
            for old_dataset, failed_samples in failures.items()
            if len(failed_samples) > 0
        ]
        if len(new_datasets) == 0 and len(failed_datasets) == 0:
            logger.warning(
                "'%s' failed on all compatible samples.",
                self.name,
            )
        return new_datasets, failed_datasets, incompatible_datasets

    def apply_to_directory(
        self,
        packed_dataset_or_dir: Path,
        output_dir: Path,
        jobs: int,
        desc: str,
    ) -> None:
        """
        Apply to all samples in a file/directory and write the results to another.

        Datasets of types unsupported by the operation will be skipped.

        Args:
            packed_dataset_or_dir: Path to a packed dataset file, or a directory
                containing packed datasets.
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.
            desc: Description for the progress bar.

        """
        # Load from `packed_dataset_or_dir`
        datasets = load_dir(packed_dataset_or_dir)
        # Apply to all compatible datasets
        results, failures, incompatible = self.apply_to_datasets(
            datasets, desc=desc, jobs=jobs
        )
        if len(incompatible) > 0:
            logger.warning(
                "'%s' is incompatible with %d datasets found at '%s'. "
                "These datasets will be omitted from the results.",
                self.name,
                len(incompatible),
                packed_dataset_or_dir,
            )
        if len(failures) > 0:
            logger.warning(
                "'%s' failed on %d samples found at '%s'. "
                "These samples will be omitted from the results.",
                self.name,
                sum(len(f.samples) for f in failures),
                packed_dataset_or_dir,
            )
        # Save any results to `output_dir`
        if len(results) == 0:
            logger.warning(
                "'%s' could not be applied to any samples found at at '%s'",
                self.name,
                packed_dataset_or_dir,
            )
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
            for dataset in results:
                dataset.save_packed(output_dir)
