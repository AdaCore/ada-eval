from abc import abstractmethod
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from ada_eval.datasets.loader import load_packed_dataset
from ada_eval.datasets.types import (
    DatasetKind,
    EvaluatedSample,
    GeneratedSample,
    Sample,
    SampleOperation,
    get_packed_dataset_files,
)


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


class GenericTool(SampleOperation):
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

    @abstractmethod
    def supported_dataset_kinds(self) -> tuple[DatasetKind]:
        pass

    def _apply_to_directory(
        self,
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
        datasets = [
            x
            for x in datasets
            if DatasetKind.from_type(x.type) in self.supported_dataset_kinds()
        ]
        if len(datasets) == 0:
            print(
                f"No datasets supported by {self.name} could be found at:",
                packed_dataset_or_dir,
            )
            return

        results = self.apply_to_datasets(datasets, desc=desc, jobs=jobs)

        output_dir.mkdir(exist_ok=True, parents=True)
        for dataset in results:
            dataset.save_packed(output_dir)


class GenerationTool(GenericTool):
    @abstractmethod
    def apply(self, sample: Sample) -> GeneratedSample:
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
            packed_dataset_or_dir=packed_dataset_or_dir,
            output_dir=output_dir,
            jobs=jobs,
            desc="Generating completions",
        )


class EvaluationTool(GenericTool):
    @abstractmethod
    def apply(self, sample: GeneratedSample) -> EvaluatedSample:
        """Evaluate a sample completion."""

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
            packed_dataset_or_dir=packed_dataset_or_dir,
            output_dir=output_dir,
            jobs=jobs,
            desc="Evaluating completions",
        )
