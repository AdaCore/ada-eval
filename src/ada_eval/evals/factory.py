from enum import Enum

from .dummy_compiler import DUMMY_COMPILER_EVAL_NAME, DummyCompiler
from .gnatprove import GNATPROVE_EVAL_NAME, GnatProve


class Eval(Enum):
    GNATPROVE = GNATPROVE_EVAL_NAME
    DUMMY_COMPILER = DUMMY_COMPILER_EVAL_NAME

    def __str__(self):
        return self.value


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> GnatProve | DummyCompiler:
    match evaluation:
        case Eval.GNATPROVE:
            return GnatProve()
        case Eval.DUMMY_COMPILER:
            return DummyCompiler()
        case _:
            raise UnsupportedEvalError(evaluation)
