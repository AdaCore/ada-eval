from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, model_serializer

from ada_eval.utils import construct_enum_case_insensitive, type_checked

from .metrics import Metric, MetricSection, metric_section, metric_value


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

    @property
    @abstractmethod
    def passed(self) -> bool:
        """Whether the evaluation stats indicate a passing result with no issues."""

    @abstractmethod
    def metrics(self, canonical_stats: EvaluationStatsBase) -> MetricSection:
        """Return a hierarchical collection of metrics for this evaluation stats."""


class EvaluationStatsInvalidBase(EvaluationStatsBase):
    passed: ClassVar = False

    def metrics(self, _: EvaluationStatsBase) -> MetricSection:
        return metric_section(sub_metrics={"evaluation errors": metric_value()})


class EvaluationStatsFailed(EvaluationStatsInvalidBase):
    exception: str


class EvaluationStatsTimedOut(EvaluationStatsInvalidBase):
    cmd_timed_out: Sequence[str]
    timeout: float


EvaluationStatsInvalid = EvaluationStatsFailed | EvaluationStatsTimedOut


class EvaluationStatsBuild(EvaluationStatsBase):
    eval: Literal[Eval.BUILD] = Eval.BUILD
    compiled: bool
    pre_format_warnings: bool
    post_format_warnings: bool

    @property
    def passed(self) -> bool:
        return self.compiled and not (
            self.pre_format_warnings or self.post_format_warnings
        )

    def metrics(self, _: EvaluationStatsBase) -> MetricSection:
        metrics: dict[str, Metric] = {}
        if self.compiled:
            if self.post_format_warnings:
                sub_metrics = {"other warnings": metric_value()}
            elif self.pre_format_warnings:
                sub_metrics = {"formatting warnings": metric_value()}
            else:
                sub_metrics = {"no warnings": metric_value()}
            metrics = {"compiled": metric_section(sub_metrics=sub_metrics)}
        else:
            metrics = {"failed to compile": metric_value()}
        return metric_section(sub_metrics=metrics)


class ProofCheck(BaseModel):
    """A check that a `SparkSample`'s solution must prove to be considered correct."""

    rule: str
    """The rule name of the check, e.g. `"VC_POSTCONDITION"`."""
    entity_name: str | None = None
    """
    Optional name of an entity to which the check must be attached.

    Includes package prefix.
    """
    src_pattern: str | None = None
    """
    An optional regex pattern that must match the source code.

    Matches starting from the check location as reported by GNATprove.
    """


class EvaluationStatsProve(EvaluationStatsBase):
    eval: Literal[Eval.PROVE] = Eval.PROVE
    result: Literal[
        "subprogram_not_found", "error", "unproved", "proved_incorrectly", "proved"
    ]
    proved_checks: Counter[str]
    unproved_checks: Counter[str]
    warnings: Counter[str]
    non_spark_entities: Sequence[str]
    missing_required_checks: Sequence[ProofCheck]
    pragma_assume_count: int
    proof_steps: int

    @property
    def passed(self) -> bool:
        return self.result == "proved"

    def metrics(self, canonical_stats: EvaluationStatsBase) -> MetricSection:
        canonical_stats = type_checked(canonical_stats, EvaluationStatsProve)
        match self.result:
            case "subprogram_not_found":
                metrics: dict[str, Metric] = {"subprogram not found": metric_value()}
            case "error" | "unproved":
                metrics = {self.result: metric_value()}
            case "proved_incorrectly":
                sub_metrics: dict[str, Metric] = {}
                for key, value in {
                    "warnings": self.warnings.total(),
                    "non-spark entities": len(self.non_spark_entities),
                    "missing required checks": len(self.missing_required_checks),
                    "pragma assume": self.pragma_assume_count,
                }.items():
                    if value > 0:
                        sub_metrics[key] = metric_value(value=value)
                metrics = {
                    "proved incorrectly": metric_section(sub_metrics=sub_metrics)
                }
            case "proved":
                sub_metrics = {
                    "total proof steps": metric_value(value=self.proof_steps)
                }
                absent = (
                    canonical_stats.unproved_checks - self.unproved_checks
                ).total()
                extra = (self.proved_checks - canonical_stats.proved_checks).total()
                for key, value in {
                    "absent checks": absent,
                    "unnecessary checks": extra,
                }.items():
                    if value > 0:
                        sub_metrics[key] = metric_value(value=value)
                metrics = {"proved correctly": metric_section(sub_metrics=sub_metrics)}
        return metric_section(sub_metrics=metrics)


class EvaluationStatsTest(EvaluationStatsBase):
    eval: Literal[Eval.TEST] = Eval.TEST
    compiled: bool
    passed_tests: bool

    @property
    def passed(self) -> bool:
        return self.compiled and self.passed_tests

    def metrics(self, _: EvaluationStatsBase) -> MetricSection:
        if self.compiled:
            if self.passed_tests:
                metrics = {"passed": metric_value()}
            else:
                metrics = {"tests failed": metric_value()}
        else:
            metrics = {"compilation failed": metric_value()}
        return metric_section(sub_metrics=metrics)


EvaluationStats = (
    EvaluationStatsBuild
    | EvaluationStatsProve
    | EvaluationStatsTest
    | EvaluationStatsInvalid
)
