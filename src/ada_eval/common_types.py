from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from dataclass_wizard import JSONWizard

# Unpacked samples, will always have one file and two dirs: "base", "solution", and "other.json"
BASE_DIR_NAME = "base"
SOLUTION_DIR_NAME = "solution"
UNIT_TEST_DIR_NAME = "unit_test"
OTHER_JSON_NAME = "other.json"
COMMENTS_FILE_NAME = "comments.md"
PROMPT_FILE_NAME = "prompt.md"
REFERENCE_ANSWER_FILE_NAME = "reference_answer.md"


# Enum that specifies the type of dataset
class DatasetType(Enum):
    ADA = "ada"
    EXPLAIN = "explain"
    SPARK = "spark"

    def __str__(self) -> str:
        return self.name


@dataclass(kw_only=True)
class Sloc:
    line: int
    column: int | None


@dataclass(kw_only=True)
class Location(JSONWizard):
    path: Path
    start: Sloc | None
    end: Sloc | None


@dataclass(kw_only=True)
class SampleTemplate:
    sources: Dict[Path, str]
    others: Dict[str, Any]


@dataclass(kw_only=True)
class ExplainSolution:
    reference_answer: str
    correct_statements: list[str]
    incorrect_statements: list[str]


@dataclass(kw_only=True)
class BaseSample(JSONWizard):
    """
    name (str): Name of the sample. Should be unique within the dataset.
    location (Location): Location of the sample. The path should be relative to
      to the sample root.
    prompt (str): Prompt for the sample
    sources (Dict[Path, str]): Source files for the sample, with the path as the
      key and the contents as the value. Will include any template files.
    canonical_solution (Any): Canonical solution for the sample. The type should
      be constrained by the subclass.
    comments (str): Any comments about the sample by the author. May be empty.
    """

    name: str
    location: Location
    prompt: str
    sources: Dict[Path, str]
    canonical_solution: Any
    comments: str


@dataclass(kw_only=True)
class AdaSample(BaseSample):
    """
    location_solution (Location | None): The act of writing the solution may
      move the area of interest. This field should be used to specify the updated
      location if needed.
    canonical_solution (Dict[Path, str]): Canonical solution for the sample. The
      path should be relative to the sample root. The values should be the
      contents of the files.
    unit_tests (Dict[Path, str]): Same structure as used for sources or
      canonical_solution. This should contain the unit tests for the sample.
    """

    location_solution: Location | None
    canonical_solution: Dict[Path, str]
    unit_tests: Dict[Path, str]


@dataclass(kw_only=True)
class ExplainSample(BaseSample):
    canonical_solution: ExplainSolution


@dataclass(kw_only=True)
class SparkSample(AdaSample):
    pass
