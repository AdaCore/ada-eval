import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generic, TypeVar

from tqdm import tqdm

from .loader import load_dir
from .types import Dataset, Sample, dataset_has_sample_type, save_to_dir

logger = logging.getLogger(__name__)


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

    @property
    @abstractmethod
    def name(self) -> str:
        """Name used to identify the operation in logs, progress bars, etc."""

    @property
    @abstractmethod
    def progress_bar_desc(self) -> str:
        """Description for the progress bar when applying to multiple samples."""

    @property
    @abstractmethod
    def type_map(self) -> dict[type[InputType], type[OutputType]]:
        """Map from input sample types to output sample types."""

    @abstractmethod
    def apply(self, sample: InputType) -> OutputType:
        """Apply the operation to a sample."""

    def apply_to_datasets(
        self, datasets: Iterable[Dataset[DatasetSampleType]], jobs: int
    ) -> tuple[
        list[Dataset[OutputType]],
        list[Dataset[InputType]],
        list[Dataset[DatasetSampleType]],
    ]:
        """
        Apply the operation to all samples in a collection of datasets.

        Args:
            datasets: Iterable of datasets to apply the operation to.
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
            if dataset_has_sample_type(inp_dataset, tuple(self.type_map.keys())):
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
            with tqdm(total=total_samples, desc=self.progress_bar_desc) as pbar:
                for future in as_completed(future_to_input.keys()):
                    dataset, sample = future_to_input[future]
                    try:
                        result = future.result()
                        dataset_results[dataset].append(result)
                    except Exception:
                        logging.exception(
                            "Error processing sample %s from dataset %s",
                            sample.name,
                            dataset.dirname(),
                        )
                        failures[dataset].append(sample)
                    finally:
                        pbar.update(1)

        # Sort samples by name within each dataset for consistent output when
        # running in parallel
        for sample_dict in (dataset_results, failures):
            for sample_list in sample_dict.values():
                sample_list.sort(key=lambda s: s.name)

        # Create new datasets for the transformed and failed samples
        new_datasets: list[Dataset[OutputType]] = [
            Dataset(
                name=old_dataset.name,
                sample_type=self.type_map[old_dataset.sample_type],
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
        return new_datasets, failed_datasets, incompatible_datasets

    def apply_to_directory(
        self,
        path: Path,
        output_dir: Path,
        jobs: int,
    ) -> None:
        """
        Apply to all samples in a file/directory and write the results to another.

        Datasets of types unsupported by the operation will be skipped.

        Args:
            path: Path to a packed dataset file, or a directory containing
                packed or unpacked dataset(s).
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.

        """
        # Load from `path`
        datasets = load_dir(path)
        # Apply to all compatible datasets
        results, failures, incompatible = self.apply_to_datasets(datasets, jobs=jobs)
        if len(incompatible) > 0:
            logger.warning(
                "'%s' is incompatible with %d datasets found at '%s'. "
                "These datasets will be omitted from the results.",
                self.name,
                len(incompatible),
                path,
            )
        if len(failures) > 0:
            logger.warning(
                "'%s' failed on %d samples found at '%s'. "
                "These samples will be omitted from the results.",
                self.name,
                sum(len(f.samples) for f in failures),
                path,
            )
        if len(results) == 0:
            logger.warning(
                "'%s' could not be applied to any samples found at '%s'.",
                self.name,
                path,
            )
        # Save any results to `output_dir`
        save_to_dir(results, output_dir)
