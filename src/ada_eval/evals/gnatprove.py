import logging
from typing import ClassVar, Literal

from ada_eval.datasets import (
    EvaluatedSparkSample,
    EvaluationStatsGnatProve,
    GeneratedSparkSample,
    SubprogramNotFoundError,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


# TODO (#2): Make this an attribute of each `SparkSample` or a command-line arg
PROVE_TIMEOUT_S = 60


class GnatProve(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
    """An evaluation that runs GNATprove and checks for any proof failures."""

    name: ClassVar[Literal["gnatprove"]] = "gnatprove"
    supported_types: ClassVar = {GeneratedSparkSample: EvaluatedSparkSample}

    def __init__(self) -> None:
        # Detect missing `gnatprove` before attempting any evaluation for a
        # cleaner error output.
        check_on_path("gnatprove")

    def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsGnatProve:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug("Evaluating %s with GNATprove in %s", sample.name, working_dir)
            # Search for the subprogram of interest.
            try:
                subp_lineno = sample.location.find_line_number(working_dir)
            except SubprogramNotFoundError:
                return EvaluationStatsGnatProve(
                    successfully_proven=False, subprogram_found=False
                )
            # Run `gnatprove`, specifying the unit and subprogram to analyze,
            # and ensuring that all kinds of proof failure yield a non-zero exit
            # code.
            #
            # TODO (#2): Scrape more detailed results from `obj/gnatprove.out`
            # and stdout/stderr.
            result, _ = run_cmd_with_timeout(
                [
                    "gnatprove",
                    "--checks-as-errors=on",
                    "--warnings=error",
                    "-k",
                    f"--limit-subp={sample.location.path.name}:{subp_lineno}",
                    str(sample.location.path),
                ],
                working_dir,
                PROVE_TIMEOUT_S,
            )
            # Return the `EvaluationStats`
            return EvaluationStatsGnatProve(
                successfully_proven=(result.returncode == 0), subprogram_found=True
            )
