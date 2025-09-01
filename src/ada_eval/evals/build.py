import logging
import subprocess
from pathlib import Path
from typing import ClassVar

from ada_eval.datasets import (
    Eval,
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsBuild,
    GeneratedAdaSample,
    GeneratedSparkSample,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


BUILD_TIMEOUT_S = 60


def _run_gprbuild(
    working_dir: Path, *, check_warnings: bool
) -> subprocess.CompletedProcess[str]:
    """Return the result of running `gprbuild`."""
    args = ["gprbuild"]
    if check_warnings:
        args.extend(["-gnatwae", "-gnatyy"])
    return run_cmd_with_timeout(args, working_dir, BUILD_TIMEOUT_S)[0]


class Build(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """An evaluation that runs GPRbuild and checks if compilation succeeds."""

    eval: ClassVar = Eval.BUILD
    supported_types: ClassVar = {
        GeneratedAdaSample: EvaluatedAdaSample,
        GeneratedSparkSample: EvaluatedSparkSample,
    }

    def __init__(self) -> None:
        # Detect missing `gprbuild` and/or `gnatformat` before attempting any
        # evaluation for a cleaner error output.
        check_on_path("gprbuild")
        check_on_path("gprclean")
        check_on_path("gnatformat")

    def evaluate(self, sample: GeneratedAdaSample) -> EvaluationStatsBuild:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug(
                "Evaluating '%s' with GPRbuild in %s", sample.name, working_dir
            )
            # Initialise values to success
            eval_stats = EvaluationStatsBuild(
                compiled=True,
                has_pre_format_compile_warnings=False,
                has_post_format_compile_warnings=False,
            )
            # Run `gprbuild` on the unformatted sources with warnings enabled
            run_cmd_with_timeout(
                ["gprclean", "-r"], working_dir, BUILD_TIMEOUT_S, check=True
            )
            result = _run_gprbuild(working_dir, check_warnings=True)
            if result.returncode == 0:
                return eval_stats  # Compiled successfully without warnings
            eval_stats.has_pre_format_compile_warnings = True
            # If that failed, format the sources with GNATformat and try again
            run_cmd_with_timeout(
                ["gnatformat"], working_dir, BUILD_TIMEOUT_S, check=True
            )
            result = _run_gprbuild(working_dir, check_warnings=True)
            if result.returncode == 0:
                return eval_stats  # Compiled with no warnings after formatting
            eval_stats.has_post_format_compile_warnings = True
            # If that still failed, try building with warnings disabled
            result = _run_gprbuild(working_dir, check_warnings=False)
            eval_stats.compiled = result.returncode == 0
            return eval_stats
