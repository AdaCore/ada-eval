from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from ada_eval.datasets.types.datasets import DatasetType
from ada_eval.datasets.types.samples import EvaluatedSample, GeneratedSample, Sample


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
    def supported_dataset_types(self) -> tuple[DatasetType]:
        pass


class GenerationTool(GenericTool):
    @abstractmethod
    def generate(self, sample_working_dir: Path, sample: Sample) -> GeneratedSample:
        """Generate a completion for a sample."""


class EvaluationTool(GenericTool):
    @abstractmethod
    def evaluate(
        self, sample_working_dir: Path, sample: GeneratedSample
    ) -> EvaluatedSample:
        """Evaluate a sample completion."""
