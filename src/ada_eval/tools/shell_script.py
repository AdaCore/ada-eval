import logging
from pathlib import Path
from typing import ClassVar, Self

from ada_eval.datasets.types import GenerationStats, SparkSample
from ada_eval.datasets.types.samples import GeneratedSparkSample, get_contents
from ada_eval.utils import run_cmd_with_timeout

from .generic_tool import BaseConfig, GenericTool

logger = logging.getLogger(__name__)


SHELL_SCRIPT_TOOL_NAME = "shell_script"


class ShellScriptConfig(BaseConfig):
    shell_script: Path  # Should be relative to the config file

    @classmethod
    def from_file(cls, config_file: Path) -> Self:
        config = super().from_file(config_file)
        # Resolve the shell script path relative to the config file's directory
        resolved_script_path = (config_file.parent / config.shell_script).resolve()
        return config.model_copy(update={"shell_script": resolved_script_path})


class ShellScript(GenericTool[ShellScriptConfig, SparkSample, GeneratedSparkSample]):
    name: ClassVar = SHELL_SCRIPT_TOOL_NAME
    type_map: ClassVar = {SparkSample: GeneratedSparkSample}
    config_type = ShellScriptConfig

    def apply(self, sample: SparkSample) -> GeneratedSparkSample:
        with sample.sources.unpacked() as sample_working_dir:
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
            generated_files = get_contents(sample_working_dir)
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
