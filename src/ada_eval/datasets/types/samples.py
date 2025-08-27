"""Common types used in the Ada Eval project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_serializer,
)

from ada_eval.datasets.utils import get_file_or_empty

from .directory_contents import DirectoryContents, get_contents_git_aware


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
CANONICAL_EVAL_KEY = "canonical_evaluation_results"


class PathMustBeRelativeError(Exception):
    def __init__(self, path: Path):
        super().__init__(f"Path '{path}' is not relative")


class SubprogramNotFoundError(ValueError):
    """Raised when a subprogram is not found in a file."""

    def __init__(self, subprogram_name: str, file_path: Path):
        super().__init__(f"Subprogram '{subprogram_name}' not found in '{file_path}'")


def find_subprogram_line(file_path: Path, subprogram_name: str) -> int:
    """
    Find the line number where a subprogram is first defined in an Ada file.

    Args:
        file_path: Path to the Ada file
        subprogram_name: Name of the subprogram to find

    Returns:
        Line number (1-based) where the subprogram is first found

    Raises:
        ValueError: If the subprogram is not found in the file

    """
    content = file_path.read_text("utf-8")
    lines = content.splitlines()

    # Compile the regex pattern once for efficiency
    pattern = re.compile(rf"\b(function|procedure)\s+{re.escape(subprogram_name)}\b")

    # Look for the first line that contains subprogram name
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            return i

    raise SubprogramNotFoundError(subprogram_name, file_path)


class Location(BaseModel):
    path: Path
    subprogram_name: str

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, value):
        path = Path(value)
        if path.is_absolute():
            raise PathMustBeRelativeError(path)
        return path

    @field_serializer("path", when_used="always")
    def serialize_path(self, path):
        return str(path)

    def find_line_number(self, base_path: Path) -> int:
        """
        Find the line number where this location's subprogram is defined.

        Args:
            base_path: Base path to resolve the relative path against

        Returns:
            Line number (1-based) where the subprogram is found

        """
        full_path = base_path / self.path
        return find_subprogram_line(full_path, self.subprogram_name)


VALID_SAMPLE_NAME_PATTERN = re.compile(r"^[\w-]+$")


class Sample(BaseModel):
    """
    Base class for samples in the dataset.

    Attributes:
        name (str): Name of the sample. Should be unique within the dataset.
        location (Location): Location of the sample. The path should be relative
            to to the sample root.
        prompt (str): Prompt for the sample
        sources (DirectoryContents): Source files for the sample.
        canonical_solution (object): Canonical solution for the sample. The type
            should be constrained by the subclass.
        canonical_evaluation_results (list[EvaluationStats]): Results of
            evaluating the canonical solution. Empty if no such evaluation has
            yet been performed.
        comments (str): Any comments about the sample by the author. May be empty.

    """

    name: str
    location: Location
    prompt: str
    sources: DirectoryContents
    canonical_solution: object
    canonical_evaluation_results: list[EvaluationStats]
    comments: str

    @field_validator("name")
    @classmethod
    def name_must_be_simple(cls, value):
        if not VALID_SAMPLE_NAME_PATTERN.match(value):
            raise InvalidSampleNameError(value)
        return value

    def working_dir_in(self, dataset_working_dir: Path) -> Path:
        """Get the working dir for this sample, given that of the parent dataset."""
        return dataset_working_dir / self.name

    def unpack(self, dataset_root: Path, other_data: dict[str, object] | None = None):
        """
        Unpack the sample in expanded form.

        Args:
            dataset_root: The dataset root directory to unpack into.
            other_data: Additional data to include in the `other.json` file.

        """
        dest_dir = self.working_dir_in(dataset_root)
        dest_dir.mkdir(exist_ok=True, parents=True)
        (dest_dir / PROMPT_FILE_NAME).write_text(self.prompt)
        (dest_dir / COMMENTS_FILE_NAME).write_text(self.comments)
        self.sources.unpack_to(dest_dir / BASE_DIR_NAME)
        other_data = {LOCATION_KEY: self.location.model_dump()} | (other_data or {})
        if len(self.canonical_evaluation_results) > 0:
            other_data[CANONICAL_EVAL_KEY] = [
                es.model_dump() for es in self.canonical_evaluation_results
            ]
        (dest_dir / OTHER_JSON_NAME).write_text(json.dumps(other_data, indent=4) + "\n")

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path) -> Self:
        other_data = json.loads(get_file_or_empty(sample_dir / OTHER_JSON_NAME))
        base_files = get_contents_git_aware(sample_dir / BASE_DIR_NAME)
        prompt = get_file_or_empty(sample_dir / PROMPT_FILE_NAME)
        comments = get_file_or_empty(sample_dir / COMMENTS_FILE_NAME)
        canonical_evaluation_results = _evaluation_results_adapter.validate_python(
            other_data.get(CANONICAL_EVAL_KEY, [])
        )
        return cls(
            name=sample_dir.name,
            location=Location.model_validate(other_data[LOCATION_KEY]),
            prompt=prompt,
            comments=comments,
            sources=base_files,
            canonical_solution=None,  # Placeholder
            canonical_evaluation_results=canonical_evaluation_results,
        )


def is_unpacked_sample(path: Path) -> bool:
    """
    Check if this is the path to a sample.

    This is the case if the dir contains an OTHER_JSON_NAME file.
    """
    other_json = path / OTHER_JSON_NAME
    return path.is_dir() and other_json.is_file()


class AdaSample(Sample):
    """
    Ada-specific sample extending the base Sample class.

    Attributes:
        canonical_solution (DirectoryContents): Canonical solution for the sample.
        unit_tests (DirectoryContents): The unit tests for the sample.

    """

    canonical_solution: DirectoryContents
    unit_tests: DirectoryContents

    def unpack(self, dataset_root: Path, other_data: dict[str, object] | None = None):
        super().unpack(dataset_root, other_data=other_data)
        dest_dir = self.working_dir_in(dataset_root)
        self.canonical_solution.unpack_to(dest_dir / SOLUTION_DIR_NAME)
        self.unit_tests.unpack_to(dest_dir / UNIT_TEST_DIR_NAME)

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        solution_files = get_contents_git_aware(sample_dir / SOLUTION_DIR_NAME)
        unit_test_files = get_contents_git_aware(sample_dir / UNIT_TEST_DIR_NAME)
        base_sample = Sample.load_unpacked_sample(sample_dir)
        return cls(
            name=base_sample.name,
            location=base_sample.location,
            prompt=base_sample.prompt,
            comments=base_sample.comments,
            sources=base_sample.sources,
            canonical_solution=solution_files,
            canonical_evaluation_results=base_sample.canonical_evaluation_results,
            unit_tests=unit_test_files,
        )


class ExplainSample(Sample):
    canonical_solution: str
    correct_statements: list[str]
    incorrect_statements: list[str]

    def unpack(self, dataset_root: Path, other_data: dict[str, object] | None = None):
        other_data = {
            CORRECT_STATEMENTS_KEY: self.correct_statements,
            INCORRECT_STATEMENTS_KEY: self.incorrect_statements,
        } | (other_data or {})
        super().unpack(dataset_root, other_data=other_data)
        dest_dir = self.working_dir_in(dataset_root)
        (dest_dir / REFERENCE_ANSWER_FILE_NAME).write_text(self.canonical_solution)

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        other_data = json.loads(get_file_or_empty(sample_dir / OTHER_JSON_NAME))
        reference_answer = get_file_or_empty(sample_dir / REFERENCE_ANSWER_FILE_NAME)
        base_sample = Sample.load_unpacked_sample(sample_dir)
        return cls(
            name=base_sample.name,
            location=base_sample.location,
            prompt=base_sample.prompt,
            comments=base_sample.comments,
            sources=base_sample.sources,
            canonical_solution=reference_answer,
            correct_statements=other_data.get(CORRECT_STATEMENTS_KEY, []),
            incorrect_statements=other_data.get(INCORRECT_STATEMENTS_KEY, []),
            canonical_evaluation_results=base_sample.canonical_evaluation_results,
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
    generated_solution: object

    def to_evaluated_sample(self) -> EvaluatedSample:
        """
        Return this sample as an evaluated sample.

        Returns this sample promoted to an `EvaluatedSample` (with an empty
        `evaluation_results`) if necessary, or unmodified if it is already an
        `EvaluatedSample`.
        """
        if isinstance(self, EvaluatedSample):
            return self
        return GENERATED_TYPE_TO_EVALUATED[type(self)](
            **self.model_dump(),  # Copy all fields from the original sample
            evaluation_results=[],
        )


class GeneratedAdaSample(AdaSample, GeneratedSample):
    # Note that `AdaSample` must be inherited before `GeneratedSample`, so that
    # the `canonical_solution` field has type `DirectoryContents`, not `object`.
    generated_solution: DirectoryContents


class GeneratedExplainSample(ExplainSample, GeneratedSample):
    # Note that `ExplainSample` must be inherited before `GeneratedSample`, so
    # the `canonical_solution` field has type `ExplainSolution`, not `object`.
    generated_solution: str


class GeneratedSparkSample(SparkSample, GeneratedAdaSample):
    pass


class EvaluationStatsBase(BaseModel):
    eval: str

    # Ensure `eval` is always serialised (so that union discrimination is
    # predictable even when `exclude_defaults=True`)
    @model_serializer(mode="wrap")
    def serialize_eval(self, next_):
        return {"eval": self.eval} | next_(self)


class EvaluationStatsFailed(EvaluationStatsBase):
    exception: str


class EvaluationStatsTimedOut(EvaluationStatsBase):
    cmd_timed_out: list[str]
    timeout: float


class EvaluationStatsBuild(EvaluationStatsBase):
    eval: Literal["build"] = "build"
    compiled: bool
    has_pre_format_compile_warnings: bool
    has_post_format_compile_warnings: bool


class EvaluationStatsProve(EvaluationStatsBase):
    eval: Literal["prove"] = "prove"
    successfully_proven: bool
    subprogram_found: bool


EvaluationStats = (
    EvaluationStatsBuild
    | EvaluationStatsProve
    | EvaluationStatsFailed
    | EvaluationStatsTimedOut
)

_evaluation_results_adapter = TypeAdapter(list[EvaluationStats])


class EvaluatedSample(GeneratedSample):
    evaluation_results: list[EvaluationStats]


class EvaluatedAdaSample(GeneratedAdaSample, EvaluatedSample):
    pass


class EvaluatedExplainSample(GeneratedExplainSample, EvaluatedSample):
    pass


class EvaluatedSparkSample(GeneratedSparkSample, EvaluatedAdaSample):
    pass


# Type mappings for promoting `Sample` -> `GeneratedSample` and
# `GeneratedSample` -> `EvaluatedSample`
BASE_TYPE_TO_GENERATED: dict[type[Sample], type[GeneratedSample]] = {
    AdaSample: GeneratedAdaSample,
    ExplainSample: GeneratedExplainSample,
    SparkSample: GeneratedSparkSample,
}
GENERATED_TYPE_TO_EVALUATED: dict[type[GeneratedSample], type[EvaluatedSample]] = {
    GeneratedAdaSample: EvaluatedAdaSample,
    GeneratedExplainSample: EvaluatedExplainSample,
    GeneratedSparkSample: EvaluatedSparkSample,
}
