from ada_eval.datasets import Eval

from .build import Build
from .prove import Prove


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> Build | Prove:
    match evaluation:
        case Eval.BUILD:
            return Build()
        case Eval.PROVE:
            return Prove()
        case _:
            raise UnsupportedEvalError(evaluation)
