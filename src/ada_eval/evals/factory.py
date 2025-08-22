from enum import Enum

from .build import Build
from .prove import Prove


class Eval(Enum):
    BUILD = Build.name
    PROVE = Prove.name

    def __str__(self):
        return self.value


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
