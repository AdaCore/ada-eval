from pathlib import Path

from .generic_tool import BaseConfig, GenericTool
from ada_eval.datasets.types import (
    DatasetType,
    ExplainSample,
    SparkSample,
    SampleResult,
)

import subprocess
from enum import Enum
from pydantic import BaseModel

from spark_assistant.types import ProveStats

import time


class SparkAssistantConfig(BaseConfig):
    threads: int
    iteration_limit: int


class SparkAssistantResult(SampleResult):
    stats: ProveStats


class SparkAssistant(GenericTool):
    config_type = SparkAssistantConfig

    def __init__(self, config: SparkAssistantConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: SparkAssistantConfig):
        return SparkAssistant(config)

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
        print(f"Applying SparkAssistant to {sample.name} in {sample_working_dir}")

        stats_file = sample_working_dir / "stats.json"
        # TODO figure out a way to capture compute usage of the spawned process
        result = subprocess.run(
            [
                "spark-assistant",
                "prove",
                "--timeout",
                str(self.config.timeout_s),
                "--threads",
                str(self.config.threads),
                "--file",
                str(sample.location.path),
                "--line",
                str(sample.location.start.line),
                "--stats_file",
                str(stats_file),
                "--iteration_limit",
                str(self.config.iteration_limit),
            ],
            cwd=sample_working_dir,
            capture_output=True,
            encoding="utf-8",
        )
        # result = SparkAssistantResult(
        #     exit_code=,
        #     stdout=,
        #     stderr=,
        #     runtime_ms=,
        #     stats=ProveStats.model_validate_json(stats_file.read_text(encoding="utf-8"))
        # )
