"""Script to unpack one or more datasets from the data folder. This creates
projects structures that are easier to work with.

This is the reverse process of the pack_datasets.py script.
"""

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ada_eval.common_types import (
    BASE_DIR_NAME,
    COMMENTS_FILE_NAME,
    OTHER_JSON_NAME,
    PROMPT_FILE_NAME,
    REFERENCE_ANSWER_FILE_NAME,
    SOLUTION_DIR_NAME,
    UNIT_TEST_DIR_NAME,
    AdaSample,
    BaseSample,
    DatasetType,
    ExplainSample,
    ExplainSolution,
    Location,
    SampleTemplate,
    SparkSample,
)

# from ada_eval.common_types import DatasetType
from ada_eval.paths import (
    COMPACTED_DATASETS_DIR,
    DATASET_TEMPLATES_DIR,
    EXPANDED_DATASETS_DIR,
)
from ada_eval.utils import get_sample_template, make_files_relative_to


class UnpackException(Exception):
    pass


@dataclass
class Args:
    src_dir: Path  # Path to dir containing unpacked dataset or datasets
    dest_dir: Path  # Path to dir containing unpacked dataset or datasets
    template_dir: Path  # Path to dir containing dataset templates


@dataclass
class UnpackedDataSetMetadata:
    dir: Path
    type: DatasetType
    sample_template: SampleTemplate


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(
        description="Pack expanded datasets into jsonl files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--src",
        type=Path,
        help="Path to packed dataset or dir of packed datasets",
        default=COMPACTED_DATASETS_DIR,
    )
    arg_parser.add_argument(
        "--dest",
        type=Path,
        help="Destination dir for unpacked datasets",
        default=EXPANDED_DATASETS_DIR,
    )
    arg_parser.add_argument(
        "--template_root",
        type=Path,
        help="Directory containing dataset templates",
        default=DATASET_TEMPLATES_DIR,
    )
    args = arg_parser.parse_args()
    return Args(src_dir=args.src, dest_dir=args.dest, template_dir=args.template_root)


def filter_template_files(
    files: dict[Path, str], sample_template: SampleTemplate
) -> list[Path]:
    """Removes template files from the list of files.

    Args:
        files (dict[Path, str]): files to filter
        dataset (UnpackedDataSetMetadata): metadata for the dataset that contains the sample

    Returns:
        dict[Path, str]: filtered files
    """
    return {
        p: files[p]
        for p in files
        if p not in sample_template.sources or files[p] != sample_template.sources[p]
    }


def is_packed_dataset(path: Path) -> bool:
    """Returns true if this is the path to a packed dataset. This is the case if
    it's a path to a jsonl file."""
    return path.is_file() and path.suffix == ".jsonl"


def is_collection_of_packed_datasets(path: Path) -> bool:
    """Returns true if this is the path to a collection of packed datasets. This
    is the case if the dir contains a jsonl file."""
    return path.is_dir() and any(is_packed_dataset(p) for p in path.iterdir())


def get_dataset_files(path: Path) -> list[Path]:
    """Returns a list of paths to the files in the dataset."""
    if is_packed_dataset(path):
        return [path]
    if is_collection_of_packed_datasets(path):
        return [p for p in path.iterdir() if is_packed_dataset(p)]
    return []


def load_ada_dataset(dataset_file: Path) -> list[AdaSample]:
    """Loads an ada dataset from a jsonl file.

    Args:
        dataset_file (Path): path to the packed dataset

    Returns:
        list[AdaSample]: list of samples in the dataset
    """
    with dataset_file.open() as f:
        return [AdaSample.from_json(line) for line in f.readlines()]


VALID_SAMPLE_NAME_CHARACTERS = re.compile(r"^[\w-]+$")


def valid_sample_name(name: str) -> bool:
    """Checks that a name only contains valid characters for a sample name"""
    return VALID_SAMPLE_NAME_CHARACTERS.match(name) is not None


def is_git_up_to_date(path: Path) -> bool:
    """Returns true if the contents of a folder are up to date in git. In this
    context we mean that no changes have been made to the files in the folder,
    including file creations/deletions/modifications."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "."],
        encoding="utf-8",
        capture_output=True,
        cwd=path,
    )
    return result.returncode == 0 and (
        result.stdout == None or result.stdout.strip() == ""
    )


def get_and_make_sample_dir(dest_dir: Path, sample: BaseSample) -> Path:
    if not valid_sample_name(sample.name):
        raise UnpackException(
            f"Invalid sample name: {sample.name}. Please only use alphanumeric characters, hyphens, and underscores."
        )
    sample_dir = dest_dir / sample.name
    sample_dir.mkdir(exist_ok=True)
    return sample_dir


def unpack_base_sample(
    sample: BaseSample, sample_dir: Path, sample_template: SampleTemplate
):
    with open(sample_dir / PROMPT_FILE_NAME, "w") as f:
        f.write(sample.prompt)
    with open(sample_dir / COMMENTS_FILE_NAME, "w") as f:
        f.write(sample.comments)
    files = filter_template_files(sample.sources, sample_template)
    for file, contents in files.items():
        src_path = sample_dir / file
        src_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, "w") as f:
            f.write(contents)
    other_json = {"location": sample.to_dict()}
    with open(sample_dir / OTHER_JSON_NAME, "w") as f:
        f.write(json.dumps(other_json, indent=4))


def unpack_ada_sample(
    sample: AdaSample, dest_dir: Path, sample_template: SampleTemplate
):
    sample_dir = get_and_make_sample_dir(dest_dir, sample, sample_template)
    unpack_base_sample(sample, sample_dir)
    other_json = {
        "location": sample.to_dict(),
        "location_solution": sample.location_solution
    }
    with open(sample_dir / OTHER_JSON_NAME, "w") as f:
        f.write(json.dumps(other_json, indent=4))



def unpack_explain_sample(
    sample: AdaSample, dest_dir: Path, sample_template: SampleTemplate
):
    sample_dir = get_and_make_sample_dir(dest_dir, sample, sample_template)
    unpack_base_sample(sample, sample_dir)


def unpack_spark_sample(
    sample: AdaSample, dest_dir: Path, sample_template: SampleTemplate
):
    unpack_ada_sample(sample, dest_dir, sample_template)


def unpack_dataset(
    dataset_file: Path, dest_root_dir: Path, sample_template: SampleTemplate
):
    """Unpacks a dataset.

    Args:
        dataset_file (Path): path to the packed dataset
        dest_root_dir (Path): path to the root directory, where the dir for the
          dataset will be created.
    """
    if not is_packed_dataset(dataset_file):
        raise ValueError(f"{dataset_file} is not a packed dataset")
    dataset_name = dataset_file.stem
    dest_dir = dest_root_dir / dataset_name
    if not is_git_up_to_date(dest_dir):
        print(
            f"There are uncommited changes in {dest_dir}. Skipping unpack of {dataset_name}"
        )
        return

    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(exist_ok=True)
    dataset_type = DatasetType(dataset_name)
    with dataset_file.open() as f:
        samples = f.readlines()
    match dataset_type:
        case DatasetType.ADA:
            samples = [AdaSample.from_json(line) for line in samples]
            unpack_sample_function = unpack_ada_sample
        case DatasetType.EXPLAIN:
            samples = [ExplainSample.from_json(line) for line in samples]
            unpack_sample_function = unpack_explain_sample
        case DatasetType.SPARK:
            samples = [SparkSample.from_json(line) for line in samples]
            unpack_sample_function = unpack_spark_sample
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    seen_samples = set()
    for sample in sample:
        if sample.name in seen_samples:
            raise UnpackException(
                f'Found duplicate sample name "{sample.name}" in "{dataset_file}"'
            )
        unpack_sample_function(sample, dest_dir, sample_template)


if __name__ == "__main__":
    args = parse_args()
    dataset_files = get_dataset_files(args.src_dir)
    sample_template = get_sample_template(args.template_dir)
    for dataset_file in dataset_files:
        unpack_dataset(dataset_file, args.dest_dir, sample_template)
