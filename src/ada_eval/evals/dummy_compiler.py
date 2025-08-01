import logging
from time import sleep
from typing import Literal

from ada_eval.datasets.types import (
    EvaluatedAdaSample,
    EvaluatedSparkSample,
    EvaluationStatsCompiler,
    GeneratedAdaSample,
    GeneratedSparkSample,
)

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


DUMMY_COMPILER_EVAL_NAME: Literal["compiler"] = "compiler"


class DummyCompiler(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
    """A dummy compiler evaluation that pretends compilation was successful."""

    def __init__(self) -> None:
        super().__init__(
            type_mapping={
                GeneratedAdaSample: EvaluatedAdaSample,
                GeneratedSparkSample: EvaluatedSparkSample,
            }
        )

    @property
    def name(self) -> str:
        return DUMMY_COMPILER_EVAL_NAME

    def evaluate(self, _: GeneratedAdaSample) -> EvaluationStatsCompiler:
        sleep(1)
        return EvaluationStatsCompiler(
            compiled=True,
            has_pre_format_compile_warnings=False,
            has_post_format_compile_warnings=False,
        )
