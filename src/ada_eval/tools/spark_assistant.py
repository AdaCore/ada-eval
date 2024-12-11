from pathlib import Path

from .generic_tool import GenericTool
from ada_eval.datasets.types import DatasetType, ExplainSample, SparkSample


class SparkAssistantConfig:
    pass


class SparkAssistant(GenericTool):
    def __init__(self, config: SparkAssistantConfig):
        self.threads = threads

    @classmethod
    def from_config(cls, config_file: Path):
        return SparkAssistant

    @property
    def name(self) -> str:
        return "spark_assistant"

    def supported_dataset_types(self) -> tuple[DatasetType]:
        return (DatasetType.SPARK, DatasetType.EXPLAIN)

    def apply(self, sample_working_dir: Path, sample: ExplainSample | SparkSample):
        match sample:
            case ExplainSample():
                self._apply_explain(sample_working_dir, sample)
            case SparkSample():
                self._apply_spark(sample_working_dir, sample)
            case _:
                raise ValueError(f"Unsupported sample type: {type(sample)}")

    def _apply_explain(self, sample_working_dir: Path, sample: ExplainSample):
        pass

    def _apply_spark(self, sample_working_dir: Path, sample: SparkSample):
        pass
