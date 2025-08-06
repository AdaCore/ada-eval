import logging
from abc import abstractmethod
from typing import Generic, TypeVar

from ada_eval.datasets import (
    EvaluatedSample,
    EvaluationStats,
    EvaluationStatsFailed,
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

    def __init__(
        self, type_mapping: dict[type[GeneratedSampleType], type[EvaluatedSampleType]]
    ) -> None:
        # Construct the full type mapping: unevaluated `GeneratedSample`s are converted
        # to the corresponding `EvaluatedSample` type, while the `EvaluatedSample`s
        # are mapped to themselves (but with additional `EvaluationStats`).
        full_type_mapping: dict[
            type[GeneratedSampleType | EvaluatedSampleType], type[EvaluatedSampleType]
        ] = {}
        for input_type, output_type in type_mapping.items():
            full_type_mapping[output_type] = output_type
            full_type_mapping[input_type] = output_type
        super().__init__(type_mapping=full_type_mapping)

    @property
    def progress_bar_desc(self) -> str:
        return f"Evaluating with {self.name}"

    @abstractmethod
    def evaluate(
        self, sample: EvaluatedSampleType | GeneratedSampleType
    ) -> EvaluationStats:
        """Evaluate a generated sample and return the resulting `EvaluationStats`."""

    def apply(
        self, sample: GeneratedSampleType | EvaluatedSampleType
    ) -> EvaluatedSampleType:
        # Promote the `GeneratedSample` to an  `EvaluatedSample` if necessary.
        evaluated_sample = sample.to_evaluated_sample()
        for t in self._type_mapping.values():  # Mypy doesn't understand `if any(...)`
            if isinstance(evaluated_sample, t):
                break
        else:
            raise WrongEvalOutputTypeError(evaluation=self, gen_sample=sample)
        # Evaluate the sample and add the results to the `EvaluatedSample`
        try:
            eval_stats = self.evaluate(sample)
        except Exception as e:
            logger.exception("Error during evaluation of %s", sample.name)
            eval_stats = EvaluationStatsFailed(eval_name=self.name, exception=repr(e))
        evaluated_sample.evaluation_results.append(eval_stats)
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
        gen_sample: GeneratedSample,
    ) -> None:
        super().__init__(
            f"Eval '{evaluation.name}' accepted a `GeneratedSample` of type "
            f"{type(gen_sample).__name__}, but the corresponding evaluated sample "
            f"type ({type(gen_sample.to_evaluated_sample()).__name__}) is not "
            "compatible with the eval's output types "
            f"({[t.__name__ for t in evaluation._type_mapping.values()]})."  # noqa: SLF001  # Error is used only by `GenericEval`
        )
