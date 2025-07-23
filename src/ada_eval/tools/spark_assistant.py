import subprocess
import time
from pathlib import Path

from spark_assistant.types import ProveStats

from ada_eval.datasets.types import (
    DatasetType,
    ExplainSample,
    SampleResult,
    SparkSample,
)
from ada_eval.datasets.types.samples import GeneratedSample, get_sample_files

from .generic_tool import BaseConfig, GenericTool


class SparkAssistantConfig(BaseConfig):
    threads: int
    iteration_limit: int


class SparkAssistantResult(SampleResult):
    tool_stats: ProveStats | None
    generated_solution: dict[Path, str]


class GeneratedSparkSample(GeneratedSample):
    sample: SparkSample
    result: SparkAssistantResult


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

    def apply(
        self, sample_working_dir: Path, sample: ExplainSample | SparkSample
    ) -> GeneratedSample:
        match sample:
            case ExplainSample():
                return self._apply_explain(sample_working_dir, sample)
            case SparkSample():
                return self._apply_spark(sample_working_dir, sample)
            case _:
                raise ValueError(f"Unsupported sample type: {type(sample)}")

    def _apply_explain(self, sample_working_dir: Path, sample: ExplainSample):
        pass

    def _apply_spark(
        self, sample_working_dir: Path, sample: SparkSample
    ) -> GeneratedSparkSample:
        print(f"Applying SparkAssistant to {sample.name} in {sample_working_dir}")

        stats_file = sample_working_dir / "stats.json"
        # TODO figure out a way to capture compute usage of the spawned process
        start = time.monotonic_ns()
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
                "--llm_provider",
                str(self.config.llm_config.provider),
                "--llm",
                str(self.config.llm_config.model),
                "--temperature",
                str(self.config.llm_config.temperature),
                "--max_input_tokens",
                str(self.config.llm_config.max_input_tokens),
                "--max_output_tokens",
                str(self.config.llm_config.max_output_tokens),
            ],
            check=False, cwd=sample_working_dir,
            capture_output=True,
            encoding="utf-8",
        )
        end = time.monotonic_ns()
        time_ms = (end - start) // 1_000_000

        tool_stats = None
        if stats_file.is_file():
            tool_stats = ProveStats.model_validate_json(
                stats_file.read_text(encoding="utf-8")
            )

        return GeneratedSparkSample(
            sample=sample,
            result=SparkAssistantResult(
                tool_stats=tool_stats,
                generated_solution=get_sample_files(sample_working_dir),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                runtime_ms=time_ms,
            ),
        )
