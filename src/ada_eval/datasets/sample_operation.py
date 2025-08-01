import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generic, TypeVar, cast

from tqdm import tqdm

from .loader import load_dir
from .types import Dataset, Sample

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
        self, datasets: Iterable[Dataset[Sample]], desc: str, jobs: int
    ) -> tuple[list[Dataset[OutputType]], list[Dataset[Sample]]]:
        """
        Apply the operation to all samples in a collection of datasets.

        Args:
            datasets: Iterable of datasets to apply the operation to.
            desc: Description for progress tracking.
            jobs: Number of parallel jobs to run.

        Returns:
            transformed_datasets: List of new datasets with samples transformed
                by the operation.
            incompatible_datasets: List of datasets that were of types unsupported
                by the operation.

        """
        # Filter by input type compatibility
        compatible_datasets: list[Dataset[InputType]] = [
            cast(Dataset[InputType], dataset)
            for dataset in datasets
            if any(issubclass(dataset.type, t) for t in self._type_mapping)
        ]
        if len(compatible_datasets) == 0:
            logger.warning("No datasets compatible with %s found.", self.name)
            return [], list(datasets)

        # Calculate total number of samples for progress tracking
        total_samples = sum(len(dataset.samples) for dataset in compatible_datasets)

        # Apply to each sample
        dataset_results: dict[Dataset[InputType], list[OutputType]] = {
            dataset: [] for dataset in compatible_datasets
        }
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all futures and create a mapping from future to dataset
            future_to_dataset: dict[Future[OutputType], Dataset[InputType]] = {}
            for dataset in compatible_datasets:
                for sample in dataset.samples:
                    supported_inputs = tuple(self._type_mapping.keys())
                    if not isinstance(sample, supported_inputs):
                        # Sanity check for `cast()` above
                        raise UnsupportedSampleTypeError(type(sample), supported_inputs)
                    future = executor.submit(self.apply, sample)
                    future_to_dataset[future] = dataset

            # Process futures as they complete with progress tracking
            with tqdm(total=total_samples, desc=desc) as pbar:
                for future in as_completed(future_to_dataset.keys()):
                    dataset = future_to_dataset[future]
                    try:
                        result = future.result()
                        dataset_results[dataset].append(result)
                    except Exception:
                        logging.exception("Error processing sample")
                    finally:
                        pbar.update(1)

        # Create new datasets with transformed samples
        new_datasets: list[Dataset[OutputType]] = []
        for old_dataset, results in dataset_results.items():
            if len(results) > 0:
                new_dataset = Dataset[OutputType](
                    name=old_dataset.name,
                    type=self._type_mapping[old_dataset.type],
                    samples=results,
                )
                new_datasets.append(new_dataset)
        incompatible_datasets = [
            dataset for dataset in datasets if dataset not in dataset_results
        ]
        return new_datasets, incompatible_datasets

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
        # Apply to all datasets
        results, _ = self.apply_to_datasets(datasets, desc=desc, jobs=jobs)
        if len(results) == 0:
            logger.warning(
                "%s failed to evaluate anything at %s", self.name, packed_dataset_or_dir
            )
            return
        # Save results to `output_dir`
        output_dir.mkdir(exist_ok=True, parents=True)
        for dataset in results:
            dataset.save_packed(output_dir)
