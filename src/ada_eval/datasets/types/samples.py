"""Common types used in the Ada Eval project."""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_serializer, field_validator

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


class Sloc(BaseModel):
    line: int
    column: int | None


class Location(BaseModel):
    path: Path
    start: Sloc | None
    end: Sloc | None

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, value):
        path = Path(value)
        if path.is_absolute():
            raise ValueError("Path must be relative")
        return path

    @field_serializer("path", when_used="always")
    def serlialize_path(self, path):
        return str(path)


class ExplainSolution(BaseModel):
    reference_answer: str
    correct_statements: list[str]
    incorrect_statements: list[str]


VALID_SAMPLE_NAME_PATTERN = re.compile(r"^[\w-]+$")


class SampleResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    runtime_ms: int
    # cpu_time  # TODO


class Sample(BaseModel):
    """
    name (str): Name of the sample. Should be unique within the dataset.
    location (Location): Location of the sample. The path should be relative
        to to the sample root.
    prompt (str): Prompt for the sample
    sources (dict[Path, str]): Source files for the sample, with the path as
        the key and the contents as the value.
    canonical_solution (Any): Canonical solution for the sample. The type
        should be constrained by the subclass.
    comments (str): Any comments about the sample by the author. May be empty.
    """

    name: str
    location: Location
    prompt: str
    sources: dict[Path, str]
    canonical_solution: Any
    comments: str

    @field_validator("name")
    @classmethod
    def name_must_be_simple(cls, value):
        if not VALID_SAMPLE_NAME_PATTERN.match(value):
            raise ValueError(
                f"Invalid sample name: {value}. Please only use "
                "alphanumeric characters, hyphens, and underscores."
            )
        return value

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
        other_json = {LOCATION_KEY: self.location.model_dump()}
        with open(dest_dir / OTHER_JSON_NAME, "w") as f:
            f.write(json.dumps(other_json, indent=4))

    def unpack_for_generation(self, sample_dir: Path):
        for file, contents in self.sources.items():
            src_path = sample_dir / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, "w") as f:
                f.write(contents)


class AdaSample(Sample):
    """
    location_solution (Location | None): The act of writing the solution may
        move the area of interest. This field should be used to specify the
        updated location if needed.
    canonical_solution (dict[Path, str]): Canonical solution for the sample.
        The path should be relative to the sample root. The values should be
        the contents of the files.
    unit_tests (dict[Path, str]): Same structure as used for sources or
        canonical_solution. This should contain the unit tests for the sample.
    """

    location_solution: Location | None
    canonical_solution: dict[Path, str]
    unit_tests: dict[Path, str]

    def unpack(self, dataset_root: Path):
        super().unpack(dataset_root)
        dest_dir = dataset_root / self.name
        location_solution = None
        if self.location_solution:
            location_solution = self.location_solution.model_dump()
        other_json = {
            LOCATION_KEY: self.location.model_dump(),
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


class ExplainSample(Sample):
    canonical_solution: ExplainSolution

    def unpack(self, dataset_root: Path):
        super().unpack(dataset_root)
        dest_dir = dataset_root / self.name
        with open(dest_dir / REFERENCE_ANSWER_FILE_NAME, "w") as f:
            f.write(self.canonical_solution.reference_answer)
        other_json = {
            CORRECT_STATEMENTS_KEY: self.canonical_solution.correct_statements,
            INCORRECT_STATEMENTS_KEY: self.canonical_solution.incorrect_statements,
        }
        with open(dest_dir / OTHER_JSON_NAME, "w") as f:
            f.write(json.dumps(other_json, indent=4))


class SparkSample(AdaSample):
    pass
