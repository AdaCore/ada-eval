from enum import Enum

from .gnatprove import GNATPROVE_TOOL_NAME, GnatProve


class Eval(Enum):
    GNATPROVE = GNATPROVE_TOOL_NAME

    def __str__(self):
        return self.value


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


_EVALS = {
    Eval.GNATPROVE: GnatProve,
}


def create_eval(evaluation: Eval) -> GnatProve:
    if evaluation not in _EVALS:
        raise UnsupportedEvalError(evaluation)
    else:
        return _EVALS[evaluation]()
