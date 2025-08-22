from enum import Enum

from .gnatprove import GnatProve
from .gprbuild import GprBuild


class Eval(Enum):
    GNATPROVE = GnatProve.name
    GPRBUILD = GprBuild.name

    def __str__(self):
        return self.value


class UnsupportedEvalError(Exception):
    def __init__(self, evaluation) -> None:
        super().__init__(f"Unsupported eval: {evaluation}")


def create_eval(evaluation: Eval) -> GnatProve | GprBuild:
    match evaluation:
        case Eval.GNATPROVE:
            return GnatProve()
        case Eval.GPRBUILD:
            return GprBuild()
        case _:
            raise UnsupportedEvalError(evaluation)
