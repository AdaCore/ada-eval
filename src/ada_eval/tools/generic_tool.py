import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from ada_eval.datasets import GeneratedSample, Sample, SampleOperation

logger = logging.getLogger(__name__)


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


BaseSampleType = TypeVar("BaseSampleType", bound=Sample)
GeneratedSampleType = TypeVar("GeneratedSampleType", bound=GeneratedSample)


class GenericTool(
    Generic[BaseSampleType, GeneratedSampleType],
    SampleOperation[BaseSampleType, GeneratedSampleType],
):
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
