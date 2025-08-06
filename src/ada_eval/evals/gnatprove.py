import logging
from typing import Literal

from ada_eval.datasets.types import (
    EvaluatedSparkSample,
    EvaluationStatsGnatProve,
    GeneratedSparkSample,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


GNATPROVE_EVAL_NAME: Literal["gnatprove"] = "gnatprove"

# TODO (#2): Make this an attribute of each `SparkSample` or a command-line arg
PROVE_TIMEOUT_S = 60


class GnatProve(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
    """An evaluation that runs GNATprove and checks for any proof failures."""

    def __init__(self) -> None:
        super().__init__(type_mapping={GeneratedSparkSample: EvaluatedSparkSample})
        # Detect missing `gnatprove` before attempting any evaluation for a
        # cleaner error output.
        check_on_path("gnatprove")

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
            result, _ = run_cmd_with_timeout(
                ["gnatprove", "--checks-as-errors=on", "--warnings=error", "-k"],
                working_dir,
                PROVE_TIMEOUT_S,
            )
            # Return a GeneratedSparkSample
            return EvaluationStatsGnatProve(
                successfully_proven=(result is not None and result.returncode == 0),
                timed_out=(result is None),
            )
