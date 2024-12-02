"""Script to pack one or more datasets from the data folder. This takes one or
more projects and packs them into a single jsonl file.

This is the reverse process of the unpack_datasets.py script.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ada_eval.common_types import (
    BASE_DIR_NAME,
    COMMENTS_FILE_NAME,
    CORRECT_STATEMENTS_KEY,
    INCORRECT_STATEMENTS_KEY,
    OTHER_JSON_NAME,
    PROMPT_FILE_NAME,
    REFERENCE_ANSWER_FILE_NAME,
    SOLUTION_DIR_NAME,
    UNIT_TEST_DIR_NAME,
    AdaSample,
    DatasetType,
    ExplainSample,
    ExplainSolution,
    Location,
    SparkSample,
)

# from ada_eval.common_types import DatasetType
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR


@dataclass
class Args:
    src_dir: Path  # Path to dir containing unpacked dataset or datasets
    dest_dir: Path  # Path to dir containing unpacked dataset or datasets


@dataclass
class UnpackedDataSetMetadata:
    dir: Path
    type: DatasetType


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(
        description="Pack expanded datasets into jsonl files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--src",
        type=Path,
        help="Source dir containing unpacked dataset or datasets",
        default=EXPANDED_DATASETS_DIR,
    )
    arg_parser.add_argument(
        "--dest",
        type=Path,
        help="Destination dir for packed dataset or datasets",
        default=COMPACTED_DATASETS_DIR,
    )
    args = arg_parser.parse_args()
    return Args(src_dir=args.src, dest_dir=args.dest)


def is_sample(path: Path) -> bool:
    """Returns true if this is the path to a sample. This is the case if the dir
    contains an OTHER_JSON_NAME file."""
    other_json = path / OTHER_JSON_NAME
    return path.is_dir() and other_json.is_file()


def is_dataset(path: Path) -> bool:
    """Returns true if this is a path to a directory that contains an unpacked
    dataset. For this to be true, at least one child dir must contain a sample."""
    return path.is_dir() and any(is_sample(d) for d in path.iterdir())


def is_collection_of_unpacked_datasets(path: Path) -> bool:
    """Return's true if this is a path to a directory that contains multiple datasets.
    Note that not all directories in this directory need to contain a dataset."""
    return path.is_dir() and any(is_dataset(d) for d in path.iterdir())


def get_dataset_type(path: Path) -> DatasetType | None:
    """Returns the type of dataset in the given path. Returns None if the path
    is not a dataset, or the type is not known."""
    if not is_dataset(path):
        return None
    dataset_name = path.name
    try:
        return DatasetType(dataset_name)
    except ValueError:
        return None


def get_file_or_empty(path: Path) -> str:
    """Returns the contents of a file, or an empty string if the file does not exist"""
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


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


def get_sample_files(root: Path) -> dict[Path, str]:
    """Returns a list of files in a directory and their contents, exluding any
    files that are ignored by git."""
    full_paths = git_ls_files(root)
    files = {p.relative_to(root): p.read_text("utf-8") for p in full_paths}
    return files


def pack_dataset(dataset: UnpackedDataSetMetadata, dest_dir: Path):
    """Packs a dataset into a jsonl file"""
    dest_file = dest_dir / f"{dataset.type.value}.jsonl"
    dest_file.write_text("")
    for sample_dir in dataset.dir.iterdir():
        if not is_sample(sample_dir):
            continue
        other_data = get_other_data(sample_dir)
        base_files = get_sample_files(sample_dir / BASE_DIR_NAME)
        prompt = get_sample_prompt(sample_dir)
        comments = get_sample_comments(sample_dir)

        match dataset.type:
            case DatasetType.ADA | DatasetType.SPARK:
                solution_files = get_sample_files(sample_dir / SOLUTION_DIR_NAME)
                unit_test_files = get_sample_files(sample_dir / UNIT_TEST_DIR_NAME)

                sample_class = (
                    AdaSample if dataset.type == DatasetType.ADA else SparkSample
                )

                location_solution = None
                if (
                    "location_solution" in other_data
                    and other_data["location_solution"]
                ):
                    location_solution = Location.from_dict(
                        other_data["location_solution"]
                    )
                sample = sample_class(
                    name=sample_dir.name,
                    location=Location.from_dict(other_data["location"]),
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
                    location=Location.from_dict(other_data["location"]),
                    prompt=prompt,
                    comments=comments,
                    sources=base_files,
                    canonical_solution=get_explain_solution(sample_dir, other_data),
                )
            case _:
                raise ValueError(f"Unknown dataset type: {dataset.type}")
        # Write the sample to the jsonl file
        with open(dest_file, "a") as f:
            f.write(sample.to_json() + "\n")


def pack_datasets(datasets: list[UnpackedDataSetMetadata], dest_dir: Path):
    """Packs each datasets into into a jsonl file"""
    dest_dir.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        pack_dataset(dataset, dest_dir)


def get_datasets(path: Path) -> list[UnpackedDataSetMetadata]:
    """Returns a list of datasets in the given path"""
    if not is_collection_of_unpacked_datasets(path) and not is_dataset(path):
        return []

    if is_dataset(path):
        dataset_type = get_dataset_type(path)
        if dataset_type is None:
            return []
        return [UnpackedDataSetMetadata(dir=path, type=dataset_type)]
    datasets = []
    for d in path.iterdir():
        if not is_dataset(d):
            continue
        dataset_type = get_dataset_type(d)
        if dataset_type is None:
            continue
        datasets.append(UnpackedDataSetMetadata(dir=d, type=dataset_type))
    return datasets


def main(args: Args):
    datasets = get_datasets(args.src_dir)
    pack_datasets(datasets, args.dest_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
