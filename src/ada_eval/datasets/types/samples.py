"""Common types used in the Ada Eval project."""

import json
import re
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory

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


class UnpackedDirectoryContextManager:
    """
    Context manager for unpacking a `DirectoryContents` to a temp directory.

    Returns the `Path` to the temp directory on entry, and cleans it up on exit.
    """

    contents: "DirectoryContents"
    temp_dir: TemporaryDirectory[str] | None = None

    def __init__(self, contents: "DirectoryContents"):
        self.contents = contents
        self.temp_dir = None

    def __enter__(self) -> Path:
        self.temp_dir = TemporaryDirectory()
        temp_dir_path = Path(self.temp_dir.__enter__())
        self.contents.unpack_to(temp_dir_path)
        return temp_dir_path

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_dir is not None:
            self.temp_dir.__exit__(exc_type, exc_value, traceback)
            self.temp_dir = None


class DirectoryContents(BaseModel):
    """
    The contents of a directory.

    Attributes:
        files (dict[Path, str]): A mapping of the files' relative paths to their
            contents.

    """

    files: dict[Path, str]

    def unpack_to(self, dest_dir: Path):
        """Unpack the contents into the specified directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)  # Should exist even if empty
        for rel_path, contents in self.files.items():
            full_path = dest_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with full_path.open("w") as f:
                f.write(contents)

    def unpacked(self) -> UnpackedDirectoryContextManager:
        """Return a context manager that unpacks the contents to a temp directory."""
        return UnpackedDirectoryContextManager(self)


def get_contents_git_aware(root: Path) -> DirectoryContents:
    """
    Return the contents of a directory.

    Will exclude any files that are ignored by git.
    """
    if not root.is_dir():
        return DirectoryContents(files={})
    full_paths = sorted(git_ls_files(root))
    files = {p.relative_to(root): p.read_text("utf-8") for p in full_paths}
    return DirectoryContents(files=files)


def get_contents(root: Path) -> DirectoryContents:
    """Return the contents of a directory."""
    if not root.is_dir():
        return DirectoryContents(files={})
    full_paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    files = {p.relative_to(root): p.read_text("utf-8") for p in full_paths}
    return DirectoryContents(files=files)


class Sample(BaseModel):
    """
    Base class for samples in the dataset.

    Attributes:
        name (str): Name of the sample. Should be unique within the dataset.
        location (Location): Location of the sample. The path should be relative
            to to the sample root.
        prompt (str): Prompt for the sample
        sources (DirectoryContents): Source files for the sample.
        canonical_solution (Any): Canonical solution for the sample. The type
            should be constrained by the subclass.
        comments (str): Any comments about the sample by the author. May be empty.

    """

    name: str
    location: Location
    prompt: str
    sources: DirectoryContents
    canonical_solution: object
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

    def unpack(self, dataset_root: Path):
        dest_dir = dataset_root / self.name
        dest_dir.mkdir(exist_ok=True, parents=True)
        (dest_dir / PROMPT_FILE_NAME).write_text(self.prompt)
        (dest_dir / COMMENTS_FILE_NAME).write_text(self.comments)
        self.sources.unpack_to(dest_dir / BASE_DIR_NAME)
        other_json = {LOCATION_KEY: self.location.model_dump()}
        (dest_dir / OTHER_JSON_NAME).write_text(json.dumps(other_json, indent=4))

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
        canonical_solution (DirectoryContents): Canonical solution for the sample.
        unit_tests (DirectoryContents): The unit tests for the sample.

    """

    location_solution: Location | None
    canonical_solution: DirectoryContents
    unit_tests: DirectoryContents

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
        (dest_dir / OTHER_JSON_NAME).write_text(json.dumps(other_json, indent=4))
        self.canonical_solution.unpack_to(dest_dir / SOLUTION_DIR_NAME)
        self.unit_tests.unpack_to(dest_dir / UNIT_TEST_DIR_NAME)

    @classmethod
    def load_unpacked_sample(cls, sample_dir: Path):
        other_data = json.loads(get_file_or_empty(sample_dir / OTHER_JSON_NAME))
        base_files = get_contents_git_aware(sample_dir / BASE_DIR_NAME)
        prompt = get_file_or_empty(sample_dir / PROMPT_FILE_NAME)
        comments = get_file_or_empty(sample_dir / COMMENTS_FILE_NAME)
        solution_files = get_contents_git_aware(sample_dir / SOLUTION_DIR_NAME)
        unit_test_files = get_contents_git_aware(sample_dir / UNIT_TEST_DIR_NAME)
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
        base_files = get_contents_git_aware(sample_dir / BASE_DIR_NAME)
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
    generated_solution: object


class GeneratedAdaSample(AdaSample, GeneratedSample):
    # Note that `AdaSample` must be before `GeneratedSample`, so that the
    # `canonical_solution` field has type `DirectoryContents`, not `object`.
    generated_solution: DirectoryContents


class GeneratedExplainSample(ExplainSample, GeneratedSample):
    # Note that `ExplainSample` must be before `GeneratedSample`, so that the
    # `canonical_solution` field has type `ExplainSolution`, not `object`.
    generated_solution: str


class GeneratedSparkSample(SparkSample, GeneratedAdaSample):
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


class EvaluationStatsSpark(EvaluationStats):
    successfully_proven: bool
    runtime_ms: int


class EvaluatedSparkSample(EvaluatedSample, GeneratedSparkSample):
    evaluation_stats: EvaluationStatsSpark
