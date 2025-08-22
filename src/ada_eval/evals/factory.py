from enum import Enum

from .build import Build
from .prove import Prove


class Eval(Enum):
    PROVE = Prove.name
    BUILD = Build.name

    def __str__(self):
        return self.value


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> Prove | Build:
    match evaluation:
        case Eval.PROVE:
            return Prove()
        case Eval.BUILD:
            return Build()
        case _:
            raise UnsupportedEvalError(evaluation)
