import logging
from time import sleep
from typing import Literal

from ada_eval.datasets.types import (
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsGprBuild,
    GeneratedAdaSample,
    GeneratedSparkSample,
)

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


DUMMY_GPRBUILD_EVAL_NAME: Literal["dummy_gprbuild"] = "dummy_gprbuild"


class DummyGprBuild(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """A dummy `gprbuild` evaluation that pretends compilation was successful."""

    def __init__(self) -> None:
        super().__init__(
            type_mapping={
                GeneratedAdaSample: EvaluatedAdaSample,
                GeneratedSparkSample: EvaluatedSparkSample,
            }
        )

    @property
    def name(self) -> str:
        return DUMMY_GPRBUILD_EVAL_NAME

    def evaluate(self, _: GeneratedAdaSample) -> EvaluationStatsGprBuild:
        sleep(1)
        return EvaluationStatsGprBuild(
            compiled=True,
            has_pre_format_compile_warnings=False,
            has_post_format_compile_warnings=False,
        )
