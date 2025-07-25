"""Common types used in the Ada Eval project."""

import json
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_serializer, field_validator

from ada_eval.datasets.utils import get_file_or_empty, git_ls_files


class InvalidSampleNameError(ValueError):
    """Raised when a sample name contains invalid characters."""

    def __init__(self, sample_name: str):
        super().__init__(
            f"Invalid sample name: '{sample_name}'. Please only use "
            "alphanumeric characters, hyphens, and underscores."
        )


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


class PathMustBeRelativeError(Exception):
    def __init__(self, path: Path):
        super().__init__(f"Path '{path}' must be relative")


class Location(BaseModel):
    path: Path
    start: Sloc | None
    end: Sloc | None

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, value):
        path = Path(value)
        if path.is_absolute():
            raise PathMustBeRelativeError(path)
        return path

    @field_serializer("path", when_used="always")
    def serlialize_path(self, path):
        return str(path)


class ExplainSolution(BaseModel):
    reference_answer: str
    correct_statements: list[str]
    incorrect_statements: list[str]


VALID_SAMPLE_NAME_PATTERN = re.compile(r"^[\w-]+$")


def get_sample_files_git_aware(root: Path) -> dict[Path, str]:
    """
    Return a list of files in a directory and their contents.

    Will exclude any files that are ignored by git.
    """
    if not root.is_dir():
        return {}
    full_paths = sorted(git_ls_files(root))
    return {p.relative_to(root): p.read_text("utf-8") for p in full_paths}


def get_sample_files(root: Path) -> dict[Path, str]:
    """Return a list of files in a directory and their contents."""
    if not root.is_dir():
        return {}
    full_paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    return {p.relative_to(root): p.read_text("utf-8") for p in full_paths}


class Sample(BaseModel):
    """
    Base class for samples in the dataset.

    Attributes:
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
            raise InvalidSampleNameError(value)
        return value

    def unpack(self, dataset_root: Path):
        dest_dir = dataset_root / self.name
        dest_dir.mkdir(exist_ok=True, parents=True)
        with (dest_dir / PROMPT_FILE_NAME).open("w") as f:
            f.write(self.prompt)
        with (dest_dir / COMMENTS_FILE_NAME).open("w") as f:
            f.write(self.comments)
        for file, contents in self.sources.items():
            src_path = dest_dir / BASE_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with src_path.open("w") as f:
                f.write(contents)
        other_json = {LOCATION_KEY: self.location.model_dump()}
        with (dest_dir / OTHER_JSON_NAME).open("w") as f:
            f.write(json.dumps(other_json, indent=4))

    def unpack_for_generation(self, sample_dir: Path):
        for file, contents in self.sources.items():
            src_path = sample_dir / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with src_path.open("w") as f:
                f.write(contents)

    @classmethod
    @abstractmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        raise NotImplementedError


class AdaSample(Sample):
    """
    Ada-specific sample extending the base Sample class.

    Attributes:
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
        with (dest_dir / OTHER_JSON_NAME).open("w") as f:
            f.write(json.dumps(other_json, indent=4))
        for file, contents in self.canonical_solution.items():
            src_path = dest_dir / SOLUTION_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with src_path.open("w") as f:
                f.write(contents)
        for file, contents in self.unit_tests.items():
            src_path = dest_dir / UNIT_TEST_DIR_NAME / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with src_path.open("w") as f:
                f.write(contents)

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        other_data = json.loads(get_file_or_empty(sample_dir / OTHER_JSON_NAME))
        base_files = get_sample_files_git_aware(sample_dir / BASE_DIR_NAME)
        prompt = get_file_or_empty(sample_dir / PROMPT_FILE_NAME)
        comments = get_file_or_empty(sample_dir / COMMENTS_FILE_NAME)
        solution_files = get_sample_files_git_aware(sample_dir / SOLUTION_DIR_NAME)
        unit_test_files = get_sample_files_git_aware(sample_dir / UNIT_TEST_DIR_NAME)
        location_solution = None
        if other_data.get(LOCATION_SOLUTION_KEY, None):
            location_solution = Location.model_validate(
                other_data[LOCATION_SOLUTION_KEY]
            )
        return cls(
            name=sample_dir.name,
            location=Location.model_validate(other_data[LOCATION_KEY]),
            location_solution=location_solution,
            prompt=prompt,
            comments=comments,
            sources=base_files,
            canonical_solution=solution_files,
            unit_tests=unit_test_files,
        )


class ExplainSample(Sample):
    canonical_solution: ExplainSolution

    def unpack(self, dataset_root: Path):
        super().unpack(dataset_root)
        dest_dir = dataset_root / self.name
        with (dest_dir / REFERENCE_ANSWER_FILE_NAME).open("w") as f:
            f.write(self.canonical_solution.reference_answer)
        other_json = {
            CORRECT_STATEMENTS_KEY: self.canonical_solution.correct_statements,
            INCORRECT_STATEMENTS_KEY: self.canonical_solution.incorrect_statements,
        }
        with (dest_dir / OTHER_JSON_NAME).open("w") as f:
            f.write(json.dumps(other_json, indent=4))

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        other_data = json.loads(get_file_or_empty(sample_dir / OTHER_JSON_NAME))
        base_files = get_sample_files_git_aware(sample_dir / BASE_DIR_NAME)
        prompt = get_file_or_empty(sample_dir / PROMPT_FILE_NAME)
        comments = get_file_or_empty(sample_dir / COMMENTS_FILE_NAME)
        reference_answer = get_file_or_empty(sample_dir / REFERENCE_ANSWER_FILE_NAME)
        return cls(
            name=sample_dir.name,
            location=Location.model_validate(other_data[LOCATION_KEY]),
            prompt=prompt,
            comments=comments,
            sources=base_files,
            canonical_solution=ExplainSolution(
                reference_answer=reference_answer,
                correct_statements=other_data[CORRECT_STATEMENTS_KEY],
                incorrect_statements=other_data[INCORRECT_STATEMENTS_KEY],
            ),
        )


class SparkSample(AdaSample):
    pass


class GenerationStats(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    runtime_ms: int
    # cpu_time  # TODO implement this


class GeneratedSample(Sample):
    generation_stats: GenerationStats
    generated_solution: Any

    @abstractmethod
    def unpack_for_evaluation(self, sample_dir: Path):
        raise NotImplementedError


class GeneratedAdaSample(GeneratedSample, SparkSample):
    generated_solution: dict[Path, str]

    def unpack_for_evaluation(self, sample_dir: Path):
        for file, contents in self.generated_solution.items():
            src_path = sample_dir / file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            with src_path.open("w") as f:
                f.write(contents)


class GeneratedExplainSample(GeneratedSample, SparkSample):
    generated_solution: str


class GeneratedSparkSample(GeneratedAdaSample):
    pass


def is_unpacked_sample(path: Path) -> bool:
    """
    Check if this is the path to a sample.

    This is the case if the dir contains an OTHER_JSON_NAME file.
    """
    other_json = path / OTHER_JSON_NAME
    return path.is_dir() and other_json.is_file()


class EvaluationStats(BaseModel):
    pass


class EvaluatedSample(GeneratedSample):
    evaluation_stats: EvaluationStats


class EvaluationStatsAda(EvaluationStats):
    compiled: bool
    has_pre_format_compile_warnings: bool
    has_post_format_compile_warnings: bool


class EvaluatedAdaSample(EvaluatedSample, GeneratedAdaSample):
    evaluation_stats: EvaluationStatsAda


class EvaluationStatsExplain(EvaluationStats):
    pass


class EvaluatedExplainSample(EvaluatedSample, GeneratedExplainSample):
    evaluation_stats: EvaluationStatsExplain


class EvaluationStatsSpark(EvaluationStatsAda):
    successfully_proven: bool


class EvaluatedSparkSample(EvaluatedSample, GeneratedSparkSample):
    evaluation_stats: EvaluationStatsSpark
