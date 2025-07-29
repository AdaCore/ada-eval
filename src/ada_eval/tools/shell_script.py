import logging
from pathlib import Path

from ada_eval.datasets.types import (
    DatasetType,
    GenerationStats,
    SparkSample,
)
from ada_eval.datasets.types.samples import (
    GeneratedSparkSample,
    Sample,
    get_sample_files,
)
from ada_eval.utils import run_cmd_with_timeout

from .generic_tool import BaseConfig, GenericTool

logger = logging.getLogger(__name__)


class ShellScriptConfig(BaseConfig):
    shell_script: Path  # Should be relative to the config file


class UnsupportedSampleTypeError(Exception):
    def __init__(self, sample_type):
        super().__init__(f"Unsupported sample type: {sample_type}")


class InvalidConfigTypeError(Exception):
    def __init__(self, expected_type, actual_type):
        super().__init__(
            f"Expected {expected_type.__name__}, got {actual_type.__name__}"
        )


class ShellScript(GenericTool):
    config_type = ShellScriptConfig

    def __init__(self, config: ShellScriptConfig):
        self.config = config

    @classmethod
    def from_config_file(cls, config_file: Path):
        config = ShellScriptConfig.model_validate_json(
            config_file.read_text(encoding="utf-8")
        )
        # Resolve the shell script path relative to the config file's directory
        resolved_script_path = (config_file.parent / config.shell_script).resolve()
        new_config = config.model_copy(update={"shell_script": resolved_script_path})
        return cls(new_config)

    @classmethod
    def from_config(cls, config: BaseConfig):
        if not isinstance(config, ShellScriptConfig):
            raise InvalidConfigTypeError(ShellScriptConfig, type(config))
        return ShellScript(config)

    @property
    def name(self) -> str:
        return "shell_script"

    def supported_dataset_types(self) -> tuple[DatasetType]:
        return (DatasetType.SPARK,)

    def apply(self, sample_working_dir: Path, sample: Sample) -> GeneratedSparkSample:
        match sample:
            case SparkSample():
                return self._apply_spark(sample_working_dir, sample)
            case _:
                raise UnsupportedSampleTypeError(type(sample))

    def _apply_spark(
        self, sample_working_dir: Path, sample: SparkSample
    ) -> GeneratedSparkSample:
        logger.debug(
            "Applying ShellScript to %s in %s", sample.name, sample_working_dir
        )
        # Run the shell script with the prompt as its argument
        result, time_ms = run_cmd_with_timeout(
            [str(self.config.shell_script), sample.prompt],
            sample_working_dir,
            self.config.timeout_s,
        )
        # Pack up the resulting files and return a GeneratedSparkSample
        generated_files = get_sample_files(sample_working_dir)
        if result is None:
            # Timed out
            generation_stats = GenerationStats(
                exit_code=124,  # Standard timeout exit code
                stdout="",
                stderr=f"Process timed out after {self.config.timeout_s} seconds",
                runtime_ms=time_ms,
            )
        else:
            generation_stats = GenerationStats(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                runtime_ms=time_ms,
            )
        return GeneratedSparkSample(
            **sample.model_dump(),  # Copy all fields from the original sample
            generated_solution=generated_files,
            generation_stats=generation_stats,
        )
