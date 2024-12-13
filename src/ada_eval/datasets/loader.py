import json
import subprocess
from pathlib import Path
from typing import Any

from ada_eval.datasets.types import (
    BASE_DIR_NAME,
    COMMENTS_FILE_NAME,
    CORRECT_STATEMENTS_KEY,
    INCORRECT_STATEMENTS_KEY,
    LOCATION_KEY,
    LOCATION_SOLUTION_KEY,
    OTHER_JSON_NAME,
    PROMPT_FILE_NAME,
    REFERENCE_ANSWER_FILE_NAME,
    SOLUTION_DIR_NAME,
    UNIT_TEST_DIR_NAME,
    AdaDataset,
    AdaSample,
    Dataset,
    DatasetType,
    ExplainDataset,
    ExplainSample,
    ExplainSolution,
    Location,
    SparkDataset,
    SparkSample,
)
from ada_eval.datasets.utils import (
    is_packed_dataset,
    is_unpacked_dataset,
    is_unpacked_sample,
)


def git_ls_files(root: Path) -> list[Path]:
    """Returns a list of files in a directory using git ls-files"""
    if not root.exists():
        return []
    result = subprocess.run(
        ["git", "ls-files", "-com", "--exclude-standard", "--deduplicate"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )

    git_files = [root / line for line in result.stdout.splitlines()]

    # We have to check that a path exists, as git ls-files will return files
    # that were previously committed but have since been deleted.
    return [path for path in git_files if path.is_file()]


def get_file_or_empty(path: Path) -> str:
    """Returns the contents of a file, or an empty string if the file does not exist"""
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def get_sample_files(root: Path) -> dict[Path, str]:
    """Returns a list of files in a directory and their contents, excluding any
    files that are ignored by git."""
    if not root.is_dir():
        return {}
    full_paths = git_ls_files(root)
    files = {p.relative_to(root): p.read_text("utf-8") for p in full_paths}
    return files


def get_sample_prompt(sample_root: Path) -> str:
    return get_file_or_empty(sample_root / PROMPT_FILE_NAME)


def get_sample_comments(sample_root: Path) -> str:
    return get_file_or_empty(sample_root / COMMENTS_FILE_NAME)


def get_explain_solution(
    sample_root: Path, other_data: dict[str, Any]
) -> ExplainSolution:
    file = sample_root / REFERENCE_ANSWER_FILE_NAME
    reference_answer = file.read_text(encoding="utf-8")
    return ExplainSolution(
        reference_answer=reference_answer,
        correct_statements=other_data[CORRECT_STATEMENTS_KEY],
        incorrect_statements=other_data[INCORRECT_STATEMENTS_KEY],
    )


def get_other_data(sample_root: Path) -> dict:
    return json.loads((sample_root / OTHER_JSON_NAME).read_text())


def load_unpacked_dataset(path: Path) -> Dataset:
    if not is_unpacked_dataset(path):
        raise ValueError(f"{path} is not an unpacked dataset")
    if "_" not in path.stem:
        raise ValueError(
            f"Expected unpacked dataset dir name to contain an underscore: {path}"
        )
    first_underscore = path.stem.index("_")
    dataset_type = DatasetType(path.stem[:first_underscore])
    dataset_name = path.stem[first_underscore + 1 :]
    samples = []
    for sample_dir in sorted(path.iterdir()):
        if not is_unpacked_sample(sample_dir):
            continue
        other_data = get_other_data(sample_dir)
        base_files = get_sample_files(sample_dir / BASE_DIR_NAME)
        prompt = get_sample_prompt(sample_dir)
        comments = get_sample_comments(sample_dir)

        match dataset_type:
            case DatasetType.ADA | DatasetType.SPARK:
                solution_files = get_sample_files(sample_dir / SOLUTION_DIR_NAME)
                unit_test_files = get_sample_files(sample_dir / UNIT_TEST_DIR_NAME)
                sample_class = (
                    AdaSample if dataset_type == DatasetType.ADA else SparkSample
                )
                location_solution = None
                if other_data.get(LOCATION_SOLUTION_KEY, None):
                    location_solution = Location.model_validate(
                        other_data[LOCATION_SOLUTION_KEY]
                    )
                sample = sample_class(
                    name=sample_dir.name,
                    location=Location.model_validate(other_data[LOCATION_KEY]),
                    location_solution=location_solution,
                    prompt=prompt,
                    comments=comments,
                    sources=base_files,
                    canonical_solution=solution_files,
                    unit_tests=unit_test_files,
                )
            case DatasetType.EXPLAIN:
                sample = ExplainSample(
                    name=sample_dir.name,
                    location=Location.model_validate(other_data[LOCATION_KEY]),
                    prompt=prompt,
                    comments=comments,
                    sources=base_files,
                    canonical_solution=get_explain_solution(sample_dir, other_data),
                )
            case _:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
        samples.append(sample)
    match dataset_type:
        case DatasetType.ADA:
            return AdaDataset(name=dataset_name, samples=samples, type=dataset_type)
        case DatasetType.EXPLAIN:
            return ExplainDataset(name=dataset_name, samples=samples, type=dataset_type)
        case DatasetType.SPARK:
            return SparkDataset(name=dataset_name, samples=samples, type=dataset_type)
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_packed_dataset(path: Path) -> Dataset:
    if not is_packed_dataset(path):
        raise ValueError(f"{path} is not a packed dataset")
    if "_" not in path.stem:
        raise ValueError(
            f"Expected packed dataset filename to contain an underscore: {path}"
        )
    first_underscore = path.stem.index("_")
    dataset_type = DatasetType(path.stem[:first_underscore])
    dataset_name = path.stem[first_underscore + 1 :]
    match dataset_type:
        case DatasetType.ADA:
            dataset_class = AdaDataset
            sample_class = AdaSample
        case DatasetType.EXPLAIN:
            dataset_class = ExplainDataset
            sample_class = ExplainSample
        case DatasetType.SPARK:
            dataset_class = SparkDataset
            sample_class = SparkSample
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    with path.open() as f:
        lines = f.readlines()
    samples = [sample_class.model_validate_json(x, strict=True) for x in lines]
    return dataset_class(name=dataset_name, samples=samples, type=dataset_type)
