"""Script to unpack one or more datasets from the data folder. This creates
projects structures that are easier to work with.

This is the reverse process of the pack_datasets.py script.
"""

import argparse
import json
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
        default=COMPACTED_DATASETS_DIR,
    )
    arg_parser.add_argument(
        "--template_root",
        type=Path,
        help="Directory containing dataset templates",
        default=DATASET_TEMPLATES_DIR,
    )
    args = arg_parser.parse_args()
    return Args(src_dir=args.src, dest_dir=args.dest, template_dir=args.template_root)


# def filter_template_files(files: list[tuple[Path, Path]], dataset: UnpackedDataSetMetadata) -> list[Path]:
#     """Removes template files from the list of files.

#     Args:
#         files (list[tuple[Path, Path]]): List of file paths. Each tuple contains the short path (relative to the sample's base dir) and the full path.
#         dataset (UnpackedDataSetMetadata): metadata for the dataset that contains the sample

#     Returns:
#         list[Path]: list of filtered full paths
#     """
#     res = []
#     for short, long in files:
#         if short not in dataset.sample_template.sources:
#             res.append(long)
#         elif long.read_text() != dataset.sample_template.sources[short]:
#             res.append(long)
#     return res

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

def unpack_dataset(dataset_file: Path, dest_root_dir: Path):
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
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(exist_ok=True)
    dataset_type = DatasetType(dataset_name)
    with dataset_file.open() as f:
        samples = f.readlines()
    match dataset_type:
        case DatasetType.ADA:
            samples = [AdaSample.from_json(line) for line in samples]
        case DatasetType.EXPLAIN:
            samples = [ExplainSample.from_json(line) for line in samples]
        case DatasetType.SPARK:
            samples = [ExplainSample.from_json(line) for line in samples]
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    print(samples)

if __name__ == "__main__":
    args = parse_args()
    dataset_files = get_dataset_files(args.src_dir)
    for dataset_file in dataset_files:
        unpack_dataset(dataset_file, args.dest_dir)
