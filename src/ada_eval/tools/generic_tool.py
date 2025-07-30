from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from ada_eval.datasets.loader import load_packed_dataset
from ada_eval.datasets.types.datasets import (
    Dataset,
    DatasetKind,
    get_packed_dataset_files,
)
from ada_eval.datasets.types.samples import EvaluatedSample, GeneratedSample, Sample


class UnsupportedSampleTypeError(TypeError):
    def __init__(self, sample_type, expected_type):
        super().__init__(
            f"Unsupported sample type: got {sample_type.__name__}, "
            f"expected {expected_type.__name__}"
        )


class LlmProvider(Enum):
    Ollama = "Ollama"
    Bedrock = "Bedrock"

    def __str__(self) -> str:
        return self.value


class LLMConfig(BaseModel):
    provider: LlmProvider
    model: str
    temperature: float = 0.8
    max_input_tokens: int = 4096
    max_output_tokens: int = 4096


class BaseConfig(BaseModel):
    timeout_s: int


class GenericTool(ABC):
    config_type: type[BaseConfig]

    @classmethod
    def from_config_file(cls, config_file: Path):
        return cls.from_config(
            cls.config_type.model_validate_json(config_file.read_text(encoding="utf-8"))
        )

    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseConfig):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def supported_dataset_kinds(self) -> tuple[DatasetKind]:
        pass

    def _apply_to_directory(
        self,
        func: Callable[[Sample], Sample],
        packed_dataset_or_dir: Path,
        output_dir: Path,
        jobs: int,
        desc: str,
    ) -> None:
        """
        Apply `func` to all samples in one file/directory and write results to another.

        Intended for use with `self.generate`, `self.evaluate` etc. as `func`,
        so checks that the datasets are in `self.supported_dataset_types()`.

        Args:
            func: Function to apply to each sample.
            packed_dataset_or_dir: Path to a packed dataset file, or a directory
                containing packed datasets.
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.
            desc: Description for the progress bar.

        """

        dataset_files = get_packed_dataset_files(packed_dataset_or_dir)
        if len(dataset_files) == 0:
            print(f"No datasets could be found at: {packed_dataset_or_dir}")
            return
        datasets = [load_packed_dataset(path) for path in dataset_files]
        datasets = [x for x in datasets if x.type in self.supported_dataset_kinds()]
        if len(datasets) == 0:
            print(
                f"No datasets supported by {self.name} could be found at:",
                packed_dataset_or_dir,
            )
            return

        # Calculate total number of samples for progress tracking
        total_samples = sum(len(dataset.samples) for dataset in datasets)

        # Apply to each sample
        dataset_results: dict[Dataset, list[Sample]] = {
            dataset: [] for dataset in datasets
        }
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all futures and create a mapping from future to dataset
            future_to_dataset: dict[Future[Sample], Dataset] = {}
            for dataset in datasets:
                for sample in dataset.samples:
                    future = executor.submit(func, sample)
                    future_to_dataset[future] = dataset

            # Process futures as they complete with progress tracking
            with tqdm(
                total=total_samples,
                desc=desc,
            ) as pbar:
                for future in as_completed(future_to_dataset.keys()):
                    dataset = future_to_dataset[future]
                    try:
                        result = future.result()
                        dataset_results[dataset].append(result)
                    except Exception as e:  # noqa: BLE001 we want to catch any and all exceptions
                        print(f"Error processing sample: {e}")
                    finally:
                        pbar.update(1)

        # Write the results to file
        output_dir.mkdir(exist_ok=True, parents=True)
        for dataset, results in dataset_results.items():
            output_file = output_dir / f"{dataset.dirname()}.jsonl"
            with output_file.open("w") as f:
                for result in results:
                    f.write(result.model_dump_json() + "\n")


class GenerationTool(GenericTool):
    @abstractmethod
    def generate(self, sample: Sample) -> GeneratedSample:
        """Generate a completion for a sample."""

    def generate_dir(
        self,
        packed_dataset_or_dir: Path,
        output_dir: Path,
        jobs: int,
    ) -> None:
        """
        Generate completions for all samples in a file or directory.

        The results are written to `output_dir`.

        Args:
            packed_dataset_or_dir: Path to a packed dataset file, or a directory
                containing packed datasets.
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.

        """
        self._apply_to_directory(
            func=self.generate,
            packed_dataset_or_dir=packed_dataset_or_dir,
            output_dir=output_dir,
            jobs=jobs,
            desc="Generating completions",
        )


class EvaluationTool(GenericTool):
    @abstractmethod
    def evaluate(self, sample: GeneratedSample) -> EvaluatedSample:
        """Evaluate a sample completion."""

    def _evaluate_with_type_check(self, sample: Sample) -> EvaluatedSample:
        if not isinstance(sample, GeneratedSample):
            raise UnsupportedSampleTypeError(type(sample), GeneratedSample)
        return self.evaluate(sample)

    def evaluate_dir(
        self,
        packed_dataset_or_dir: Path,
        output_dir: Path,
        jobs: int,
    ) -> None:
        """
        Evaluate completions for all samples in a file or directory.

        The results are written to `output_dir`.

        Args:
            packed_dataset_or_dir: Path to a packed dataset file, or a directory
                containing packed datasets.
            output_dir: Directory where the results will be saved.
            jobs: Number of parallel jobs to run.

        Raises:
            UnsupportedSampleTypeError: If a sample is not a `GeneratedSample`.

        """
        self._apply_to_directory(
            func=self._evaluate_with_type_check,
            packed_dataset_or_dir=packed_dataset_or_dir,
            output_dir=output_dir,
            jobs=jobs,
            desc="Evaluating completions",
        )
