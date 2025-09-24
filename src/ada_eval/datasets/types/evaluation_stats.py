from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_serializer

from ada_eval.utils import construct_enum_case_insensitive


class Eval(StrEnum):
    BUILD = "build"
    PROVE = "prove"
    TEST = "test"

    # Constructor should be case-insensitive
    @classmethod
    def _missing_(cls, value):
        return construct_enum_case_insensitive(cls, value)


class EvaluationStatsBase(BaseModel):
    eval: Eval

    # Ensure `eval` is always serialised (so that union discrimination is
    # predictable even when `exclude_defaults=True`)
    @model_serializer(mode="wrap")
    def serialize_eval(self, next_):
        return {"eval": str(self.eval)} | next_(self)


class EvaluationStatsFailed(EvaluationStatsBase):
    exception: str


class EvaluationStatsTimedOut(EvaluationStatsBase):
    cmd_timed_out: Sequence[str]
    timeout: float


class EvaluationStatsBuild(EvaluationStatsBase):
    eval: Literal[Eval.BUILD] = Eval.BUILD
    compiled: bool
    pre_format_warnings: bool
    post_format_warnings: bool


class EvaluationStatsProve(EvaluationStatsBase):
    eval: Literal[Eval.PROVE] = Eval.PROVE
    successfully_proven: bool
    subprogram_found: bool


class EvaluationStatsProve_New(EvaluationStatsBase):  # noqa: N801
    eval: Literal[Eval.PROVE] = Eval.PROVE
    result: Literal[
        "subprogram_not_found", "error", "unproved", "proved_incorrectly", "proved"
    ]
    proved_checks: Counter[str]
    unproved_checks: Counter[str]
    warnings: Counter[str]
    missing_required_checks: int
    pragma_assume_count: int
    proof_steps: int


class EvaluationStatsTest(EvaluationStatsBase):
    eval: Literal[Eval.TEST] = Eval.TEST
    compiled: bool
    passed_tests: bool


EvaluationStats = (
    EvaluationStatsBuild
    | EvaluationStatsProve
    | EvaluationStatsProve_New
    | EvaluationStatsTest
    | EvaluationStatsFailed
    | EvaluationStatsTimedOut
)
