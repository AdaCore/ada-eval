"""Common types used in the Ada Eval project."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from dataclass_wizard import JSONWizard  # type: ignore

# Unpacked samples should always have at least:
# - one file: "other.json"
# - two dirs: "base", "solution"
BASE_DIR_NAME = "base"
SOLUTION_DIR_NAME = "solution"
UNIT_TEST_DIR_NAME = "unit_test"
OTHER_JSON_NAME = "other.json"
COMMENTS_FILE_NAME = "comments.md"
PROMPT_FILE_NAME = "prompt.md"
REFERENCE_ANSWER_FILE_NAME = "reference_answer.md"
CORRECT_STATEMENTS_KEY = "correct_statements"
INCORRECT_STATEMENTS_KEY = "incorrect_statements"
LOCATION_KEY = "location"
LOCATION_SOLUTION_KEY = "location_solution"


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


VALID_SAMPLE_NAME_PATTERN = re.compile(r"^[\w-]+$")


@dataclass(kw_only=True)
class BaseSample(JSONWizard):
    """
    name (str): Name of the sample. Should be unique within the dataset.
    location (Location): Location of the sample. The path should be relative
        to to the sample root.
    prompt (str): Prompt for the sample
    sources (Dict[Path, str]): Source files for the sample, with the path as
        the key and the contents as the value. Will include any template files.
    canonical_solution (Any): Canonical solution for the sample. The type
        should be constrained by the subclass.
    comments (str): Any comments about the sample by the author. May be empty.
    """

    name: str
    location: Location
    prompt: str
    sources: Dict[Path, str]
    canonical_solution: Any
    comments: str

    def __post_init__(self):
        if not VALID_SAMPLE_NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Invalid sample name: {self.name}. Please only use "
                "alphanumeric characters, hyphens, and underscores."
            )

    def unpack(self, dataset_root: Path):
        dest_dir = dataset_root / self.name
        dest_dir.mkdir(exist_ok=True, parents=True)
        with open(dest_dir / PROMPT_FILE_NAME, "w") as f:
            f.write(self.prompt)
        with open(dest_dir / COMMENTS_FILE_NAME, "w") as f:
            f.write(self.comments)
        for file, contents in self.sources.items():
            src_path = dest_dir / BASE_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, "w") as f:
                f.write(contents)
        other_json = {LOCATION_KEY: self.location.to_dict()}
        with open(dest_dir / OTHER_JSON_NAME, "w") as f:
            f.write(json.dumps(other_json, indent=4))

    def unpack_for_generation(self, sample_dir: Path):
        for file, contents in self.sources.items():
            src_path = sample_dir / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, "w") as f:
                f.write(contents)


@dataclass(kw_only=True)
class AdaSample(BaseSample):
    """
    location_solution (Location | None): The act of writing the solution may
        move the area of interest. This field should be used to specify the
        updated location if needed.
    canonical_solution (Dict[Path, str]): Canonical solution for the sample.
        The path should be relative to the sample root. The values should be
        the contents of the files.
    unit_tests (Dict[Path, str]): Same structure as used for sources or
        canonical_solution. This should contain the unit tests for the sample.
    """

    location_solution: Location | None
    canonical_solution: Dict[Path, str]
    unit_tests: Dict[Path, str]

    def unpack(self, dataset_root: Path):
        super().unpack(dataset_root)
        dest_dir = dataset_root / self.name
        location_solution = None
        if self.location_solution:
            location_solution = self.location_solution.to_dict()
        other_json = {
            LOCATION_KEY: self.location.to_dict(),
            LOCATION_SOLUTION_KEY: location_solution,
        }
        with open(dest_dir / OTHER_JSON_NAME, "w") as f:
            f.write(json.dumps(other_json, indent=4))
        for file, contents in self.canonical_solution.items():
            src_path = dest_dir / SOLUTION_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, "w") as f:
                f.write(contents)
        for file, contents in self.unit_tests.items():
            src_path = dest_dir / UNIT_TEST_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, "w") as f:
                f.write(contents)


@dataclass(kw_only=True)
class ExplainSample(BaseSample):
    canonical_solution: ExplainSolution

    def unpack(self, dataset_root: Path):
        super().unpack(dataset_root)
        dest_dir = dataset_root / self.name
        with open(dest_dir / REFERENCE_ANSWER_FILE_NAME, "w") as f:
            f.write(self.solution.reference_answer)
        other_json = {
            CORRECT_STATEMENTS_KEY: self.solution.correct_statements,
            INCORRECT_STATEMENTS_KEY: self.solution.incorrect_statements,
        }
        with open(dest_dir / OTHER_JSON_NAME, "w") as f:
            f.write(json.dumps(other_json, indent=4))


@dataclass(kw_only=True)
class SparkSample(AdaSample):
    pass
