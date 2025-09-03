import logging
import subprocess
from abc import abstractmethod
from typing import Generic, TypeVar

from ada_eval.datasets import (
    EVALUATED_SAMPLE_TYPES,
    Eval,
    EvaluatedSample,
    EvaluationStats,
    EvaluationStatsFailed,
    EvaluationStatsTimedOut,
    GeneratedSample,
    SampleOperation,
)

logger = logging.getLogger(__name__)


GeneratedSampleType = TypeVar("GeneratedSampleType", bound=GeneratedSample)
EvaluatedSampleType = TypeVar("EvaluatedSampleType", bound=EvaluatedSample)


class GenericEval(
    Generic[GeneratedSampleType, EvaluatedSampleType],
    SampleOperation[GeneratedSampleType | EvaluatedSampleType, EvaluatedSampleType],
):
    """
    An abstract base class for evaluations.

    Converts `GeneratedSample` to `EvaluatedSample`, or extends the `evaluation_stats`
    of an existing `EvaluatedSample`.
    """

    @property
    @abstractmethod
    def eval(self) -> Eval:
        """The `Eval` this evaluation implements."""

    @property
    @abstractmethod
    def supported_types(
        self,
    ) -> dict[type[GeneratedSampleType], type[EvaluatedSampleType]]:
        """The supported `GeneratedSample`s (and corresponding `EvaluatedSample`s)."""

    @abstractmethod
    def evaluate(
        self, sample: EvaluatedSampleType | GeneratedSampleType
    ) -> EvaluationStats:
        """Evaluate a generated sample and return the resulting `EvaluationStats`."""

    @property
    def name(self) -> str:
        return self.eval

    @property
    def progress_bar_desc(self) -> str:
        return f"Evaluating with {self.eval}"

    @property
    def type_map(
        self,
    ) -> dict[
        type[GeneratedSampleType | EvaluatedSampleType], type[EvaluatedSampleType]
    ]:
        # Construct the full type mapping: unevaluated `GeneratedSample`s are
        # converted to the corresponding `EvaluatedSample` type, while the
        # `EvaluatedSample`s are mapped to themselves.
        return self.supported_types | {
            output_type: output_type for output_type in self.supported_types.values()
        }

    def apply(
        self, sample: GeneratedSampleType | EvaluatedSampleType
    ) -> EvaluatedSampleType:
        # Promote the `GeneratedSample` to an  `EvaluatedSample` if necessary.
        if isinstance(sample, EvaluatedSample):
            # Keep any prior `evaluation_results`, but avoid mutating the input.
            evaluated_sample: EvaluatedSample = sample.model_copy()
        else:
            evaluated_sample = EVALUATED_SAMPLE_TYPES[sample.kind](
                **sample.model_dump(),  # Copy all fields from the original sample
                evaluation_results=[],
            )
        # Python's type annotations support neither type maps nor intersection
        # types, so we can only check that `self.type_map` is correctly
        # configured at runtime.
        if not isinstance(evaluated_sample, self.type_map[type(sample)]):
            raise WrongEvalOutputTypeError(
                evaluation=self, generated=sample, evaluated=evaluated_sample
            )
        # Evaluate the sample
        try:
            eval_stats = self.evaluate(sample)
        except subprocess.TimeoutExpired as e:
            logger.warning(
                "Evaluation of sample %s failed due to subprocess timeout (%s seconds)",
                sample.name,
                e.timeout,
            )
            eval_stats = EvaluationStatsTimedOut(
                eval=self.name, cmd_timed_out=e.cmd, timeout=e.timeout
            )
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError):
                e.add_note(f"stdout: {e.stdout!r}")
                e.add_note(f"stderr: {e.stderr!r}")
            logger.exception("Error during evaluation of sample %s", sample.name)
            eval_stats = EvaluationStatsFailed(eval=self.eval, exception=repr(e))
        # Add the results to the `EvaluatedSample` (append without mutating)
        evaluated_sample.evaluation_results = [
            *evaluated_sample.evaluation_results,
            eval_stats,
        ]
        return evaluated_sample


class WrongEvalOutputTypeError(TypeError):
    """
    Raised when an eval has the wrong output type.

    Specifically, raised when an input `GeneratedSample`'s corresponding
    `EvaluatedSample` (from `to_evaluated_sample()`) is not compatible with the
    eval's `output_type`. This should only happen if the eval's `type_mapping`
    is misconfigured.
    """

    def __init__(
        self,
        evaluation: GenericEval[GeneratedSampleType, EvaluatedSampleType],
        generated: GeneratedSampleType | EvaluatedSampleType,
        evaluated: EvaluatedSample,
    ) -> None:
        super().__init__(
            f"Eval '{evaluation.name}' purports to map samples of type "
            f"{type(generated).__name__} to type "
            f"{evaluation.type_map[type(generated)].__name__}, "
            f"but the corresponding evaluated type is actually "
            f"{type(evaluated).__name__}."
        )
