import logging
from typing import Generic, TypeVar

from ada_eval.datasets import EvaluatedSample, GeneratedSample, SampleOperation

logger = logging.getLogger(__name__)


GeneratedSampleType = TypeVar("GeneratedSampleType", bound=GeneratedSample)
EvaluatedSampleType = TypeVar("EvaluatedSampleType", bound=EvaluatedSample)


class GenericEval(
    Generic[GeneratedSampleType, EvaluatedSampleType],
    SampleOperation[GeneratedSampleType, EvaluatedSampleType],
):
    pass
