from enum import Enum

from .dummy_gprbuild import DUMMY_GPRBUILD_EVAL_NAME, DummyGprBuild
from .gnatprove import GNATPROVE_EVAL_NAME, GnatProve


class Eval(Enum):
    GNATPROVE = GNATPROVE_EVAL_NAME
    DUMMY_GPRBUILD = DUMMY_GPRBUILD_EVAL_NAME

    def __str__(self):
        return self.value


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> GnatProve | DummyGprBuild:
    match evaluation:
        case Eval.GNATPROVE:
            return GnatProve()
        case Eval.DUMMY_GPRBUILD:
            return DummyGprBuild()
        case _:
            raise UnsupportedEvalError(evaluation)
