import logging
import os
from typing import ClassVar

from ada_eval.datasets import (
    Eval,
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsTest,
    GeneratedAdaSample,
    GeneratedSparkSample,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


BUILD_TIMEOUT_S = 60

PROJ_ARG = "-Ptests.gpr"


class Test(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """An evaluation that builds and runs the unit tests for Ada and SPARK samples."""

    eval: ClassVar = Eval.TEST
    supported_types: ClassVar = {
        GeneratedAdaSample: EvaluatedAdaSample,
        GeneratedSparkSample: EvaluatedSparkSample,
    }

    def __init__(self) -> None:
        # Detect missing `gprbuild` and/or `gprclean` before attempting any
        # evaluation for a cleaner error output.
        check_on_path("gprbuild")
        check_on_path("gprclean")

    def evaluate(self, sample: GeneratedAdaSample) -> EvaluationStatsTest:
        with (
            sample.generated_solution.unpacked() as src_working_dir,
            sample.unit_tests.unpacked() as test_working_dir,
        ):
            env = os.environ.copy()
            env["GPR_PROJECT_PATH"] = str(src_working_dir)
            logger.debug(
                "Evaluating '%s' in '%s' with unit tests in %s",
                sample.name,
                src_working_dir,
                test_working_dir,
            )
            # Build the tests
            run_cmd_with_timeout(
                ["gprclean", "-r", PROJ_ARG],
                test_working_dir,
                BUILD_TIMEOUT_S,
                check=True,
                env=env,
            )
            (result, _) = run_cmd_with_timeout(
                ["gprbuild", PROJ_ARG],
                test_working_dir,
                BUILD_TIMEOUT_S,
                env=env,
            )
            if result.returncode != 0:
                # Failed to compile the tests
                return EvaluationStatsTest(compiled=False, passed_tests=False)
            # Run the tests
            (result, _) = run_cmd_with_timeout(
                ["./bin/tests"],
                test_working_dir,
                BUILD_TIMEOUT_S,
            )
            if result.returncode == 0:
                return EvaluationStatsTest(compiled=True, passed_tests=True)
            return EvaluationStatsTest(compiled=True, passed_tests=False)
