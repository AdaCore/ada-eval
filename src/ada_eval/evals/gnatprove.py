import logging
import subprocess
from typing import Literal

from ada_eval.datasets.types import (
    EvaluatedSparkSample,
    EvaluationStatsGnatProve,
    GeneratedSparkSample,
)
from ada_eval.utils import ExecutableNotFoundError, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


GNATPROVE_EVAL_NAME: Literal["GNATprove"] = "GNATprove"

# TODO (#2): Make this an attribute of each `SparkSample` or a command-line arg
PROVE_TIMEOUT_S = 60


class GnatProve(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
    """An evaluation that runs GNATprove and checks for any proof failures."""

    def __init__(self) -> None:
        super().__init__(type_mapping={GeneratedSparkSample: EvaluatedSparkSample})
        # Check `gnatprove` is available in the PATH
        try:
            subprocess.run(["gnatprove", "--version"], capture_output=True)  # noqa: PLW1510  # `check` is irrelevant here
        except FileNotFoundError as e:
            raise ExecutableNotFoundError("gnatprove") from e

    @property
    def name(self) -> str:
        return GNATPROVE_EVAL_NAME

    def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsGnatProve:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug("Evaluating %s with GNATprove in %s", sample.name, working_dir)
            # Run `gnatprove` with no arguments (except those required to get
            # non-zero exit code on failure)
            #
            # TODO (#2): Restrict which subprogram is analyzed and scrape more
            # detailed results from `obj/gnatprove.out`
            result, time_ms = run_cmd_with_timeout(
                ["gnatprove", "--checks-as-errors=on", "--warnings=error", "-k"],
                working_dir,
                PROVE_TIMEOUT_S,
            )
            # Return a GeneratedSparkSample
            successfully_proven = False if result is None else (result.returncode == 0)
            return EvaluationStatsGnatProve(
                successfully_proven=successfully_proven, runtime_ms=time_ms
            )
