import logging
from time import sleep
from typing import ClassVar, Literal

from ada_eval.datasets import (
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsGprBuild,
    GeneratedAdaSample,
    GeneratedSparkSample,
)

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


class DummyGprBuild(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """A dummy `gprbuild` evaluation that pretends compilation was successful."""

    name: ClassVar[Literal["dummy_gprbuild"]] = "dummy_gprbuild"
    supported_types: ClassVar = {
        GeneratedAdaSample: EvaluatedAdaSample,
        GeneratedSparkSample: EvaluatedSparkSample,
    }

    def evaluate(self, _: GeneratedAdaSample) -> EvaluationStatsGprBuild:
        sleep(1)
        return EvaluationStatsGprBuild(
            compiled=True,
            has_pre_format_compile_warnings=False,
            has_post_format_compile_warnings=False,
            timed_out=False,
        )
