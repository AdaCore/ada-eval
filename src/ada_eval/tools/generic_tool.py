import logging
from enum import Enum
from pathlib import Path
from typing import Self

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

    @classmethod
    def from_file(cls, config_file: Path) -> Self:
        """Load the configuration from a JSON file."""
        return cls.model_validate_json(config_file.read_text(encoding="utf-8"))


class GenericTool[
    ConfigType: BaseConfig,
    BaseSampleType: Sample,
    GeneratedSampleType: GeneratedSample,
](
    SampleOperation[BaseSampleType, GeneratedSampleType],
):
    config: ConfigType

    # `config_type` is a class variable, but cannot be typed as such because a
    # `TypeVar` cannot be used in a `ClassVar`, and class properties are
    # deprecated.
    config_type: type[ConfigType]

    def __init__(self, config: ConfigType):
        self.config = config

    @property
    def progress_bar_desc(self) -> str:
        return f"Generating completions with {self.name}"

    @classmethod
    def from_config_file(cls, config_file: Path) -> Self:
        return cls(cls.config_type.from_file(config_file))
