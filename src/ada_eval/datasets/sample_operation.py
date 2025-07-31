import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generic, TypeVar, cast

from tqdm import tqdm

from .loader import load_packed_dataset
from .types import Dataset, Sample, UnsupportedSampleTypeError, get_packed_dataset_files

logger = logging.getLogger(__name__)


InputType = TypeVar("InputType", bound=Sample)
OutputType = TypeVar("OutputType", bound=Sample)


class SampleOperation(ABC, Generic[InputType, OutputType]):
    """An operation that converts one type of `Sample` to another."""

    input_type: type[InputType]
    output_type: type[OutputType]

    def __init__(self, input_type: type[InputType], output_type: type[OutputType]):
        self.input_type = input_type
        self.output_type = output_type

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, sample: InputType) -> OutputType:
        """Apply the operation to a sample."""

    def apply_to_datasets(
        self, datasets: Iterable[Dataset[Sample]], desc: str, jobs: int
    ) -> list[Dataset[OutputType]]:
        """
        Apply the operation to all samples in a collection of datasets.

        Returns a list of new datasets with samples transformed by the operation,
        omitting any datasets that are not of a type compatible with the operation.

        Args:
            datasets: Iterable of datasets to apply the operation to.
            desc: Description for progress tracking.
            jobs: Number of parallel jobs to run.

        """
        # Filter by input type compatibility
        compatible_datasets: list[Dataset[InputType]] = [
            cast(Dataset[InputType], dataset)
            for dataset in datasets
            if issubclass(dataset.type, self.input_type)
        ]
        if len(compatible_datasets) == 0:
            logger.warning("No datasets compatible with %s found.", self.name)
            return []

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
                    if not isinstance(sample, self.input_type):
                        # Sanity check for `cast()` above
                        raise UnsupportedSampleTypeError(type(sample), self.input_type)
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
        for dataset, results in dataset_results.items():
            if len(results) > 0:
                new_dataset = Dataset[OutputType](
                    name=dataset.name,
                    type=self.output_type,
                    samples=results,
                )
                new_datasets.append(new_dataset)
        return new_datasets

    def apply_to_directory(
        self,
        packed_dataset_or_dir: Path,
        output_dir: Path,
        jobs: int,
        desc: str,
    ) -> None:
        """
        Apply to all samples in a file/directory and write the results to another.

        Args:
            packed_dataset_or_dir: Path to a packed dataset file, or a directory
                containing packed datasets.
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.
            desc: Description for the progress bar.

        """
        # Load from `packed_dataset_or_dir`
        dataset_files = get_packed_dataset_files(packed_dataset_or_dir)
        if len(dataset_files) == 0:
            logger.warning("No datasets could be found at: %s", packed_dataset_or_dir)
        datasets = [load_packed_dataset(path) for path in dataset_files]
        # Apply to all datasets
        results = self.apply_to_datasets(datasets, desc=desc, jobs=jobs)
        if len(results) == 0:
            logger.warning(
                "%s failed to evaluate anything at %s", self.name, packed_dataset_or_dir
            )
            return
        # Save results to `output_dir`
        output_dir.mkdir(exist_ok=True, parents=True)
        for dataset in results:
            dataset.save_packed(output_dir)
