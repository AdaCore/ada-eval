import logging
import subprocess
from pathlib import Path
from typing import ClassVar, Literal

from ada_eval.datasets import (
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsGprBuild,
    GeneratedAdaSample,
    GeneratedSparkSample,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


BUILD_TIMEOUT_S = 15


class GnatFormatError(RuntimeError):
    """Raised when `gnatformat` fails when formatting the sources."""

    def __init__(self, result: subprocess.CompletedProcess[str] | Literal["timeout"]):
        if result == "timeout":
            super().__init__("GNATformat timed out.")
        else:
            super().__init__(
                f"GNATformat failed with exit code {result.returncode}:\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )


def _run_gprbuild(
    working_dir: Path, *, check_warnings: bool
) -> subprocess.CompletedProcess[str]:
    """Return the result of running `gprbuild`."""
    args = ["gprbuild", "-f"]
    if check_warnings:
        args.extend(["-gnatwae", "-gnatyy"])
    return run_cmd_with_timeout(args, working_dir, BUILD_TIMEOUT_S)[0]


class GprBuild(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """An evaluation that runs GPRbuild and checks if compilation succeeds."""

    name: ClassVar[Literal["gprbuild"]] = "gprbuild"
    supported_types: ClassVar = {
        GeneratedAdaSample: EvaluatedAdaSample,
        GeneratedSparkSample: EvaluatedSparkSample,
    }

    def __init__(self) -> None:
        # Detect missing `gprbuild` and/or `gnatformat` before attempting any
        # evaluation for a cleaner error output.
        check_on_path("gprbuild")
        check_on_path("gnatformat")

    def evaluate(self, sample: GeneratedAdaSample) -> EvaluationStatsGprBuild:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug("Evaluating %s with GPRbuild in %s", sample.name, working_dir)
            # Initialise values to success
            eval_stats = EvaluationStatsGprBuild(
                compiled=True,
                has_pre_format_compile_warnings=False,
                has_post_format_compile_warnings=False,
            )
            # Run `gprbuild` on the unformatted sources with warnings enabled
            result = _run_gprbuild(working_dir, check_warnings=True)
            if result.returncode == 0:
                return eval_stats  # Compiled successfully without warnings
            eval_stats.has_pre_format_compile_warnings = True
            # If that failed, format the sources with GNATformat and try again
            try:
                gnatformat_result, _ = run_cmd_with_timeout(
                    ["gnatformat"], working_dir=working_dir, timeout=BUILD_TIMEOUT_S
                )
            except subprocess.TimeoutExpired as e:
                raise GnatFormatError("timeout") from e
            if gnatformat_result.returncode != 0:
                raise GnatFormatError(gnatformat_result)
            result = _run_gprbuild(working_dir, check_warnings=True)
            if result.returncode == 0:
                return eval_stats  # Compiled with no warnings after formatting
            eval_stats.has_post_format_compile_warnings = True
            # If that still failed, try building with warnings disabled
            result = _run_gprbuild(working_dir, check_warnings=False)
            eval_stats.compiled = result.returncode == 0
            return eval_stats
