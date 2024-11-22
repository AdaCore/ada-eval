"""Script to pack one or more datasets from the data folder. This takes one or
more projects and packs them into a single jsonl file.

This is the reverse process of the unpack_datasets.py script.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# from ada_eval.common_types import DatasetType
from src.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR, DATASET_TEMPLATES_DIR
from src.common_types import DatasetType

@dataclass
class Args:
    src_dir: Path  # Path to dir containing unpacked dataset or datasets
    dest_dir: Path  # Path to dir containing unpacked dataset or datasets
    template_dir: Path  # Path to dir containing dataset templates

def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--src", type=Path, help="Source dir containing unpacked dataset or datasets", default=EXPANDED_DATASETS_DIR)
    arg_parser.add_argument("--dest", type=Path, help="Destination dir for packed dataset or datasets", default=COMPACTED_DATASETS_DIR)
    arg_parser.add_argument("--template_root", type=Path, help="Directory containing dataset templates", default=DATASET_TEMPLATES_DIR)
    args = arg_parser.parse_args()
    return Args(src_dir=args.src, dest_dir=args.dest, template_dir=args.template_root)

def is_sample(path: Path) -> bool:
    """Returns true if this is the path to a sample. This is the case if the dir
    contains an "other.jsonl" file."""
    other_jsonl = path / "other.jsonl"
    return path.is_dir() and other_jsonl.is_file()

def is_dataset(path: Path) -> bool:
    """Returns true if this is a path to a directory that contains an unpacked
    dataset. For this to be true, at least one child dir must contain a sample."""
    return path.is_dir() and any(is_sample(d) for d in path.iterdir())

def is_collection_of_datasets(path: Path) -> bool:
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

def get_template_dir(template_root: Path, dataset_type: DatasetType) -> Path:
    """Returns the path the the template dir for a given dataset"""
    return template_root / dataset_type.value

def pack_dataset(dataset_dir: Path, template_dir: Path):
    """Packs a dataset into a jsonl file"""
    if not is_dataset(path):
        return
    for sample in path.iterdir():
        if not is_sample(sample):
            continue
        with open(sample / "other.jsonl") as f:
            other_data = json.load(f)
        with open(path / f"{sample.name}.jsonl", "w") as f:
            json.dump(data, f)

def pack_collection_of_datasets(path: Path, template_root: Path):
    """Packs a collection of datasets, packing each dataset into a jsonl file"""
    if not is_collection_of_datasets(path):
        return
    for dataset_dir in path.iterdir():
        if not is_dataset(dataset_dir):
            continue
        dataset_type = get_dataset_type(dataset_dir)
        if dataset_type is None:
            continue
        template_dir = get_template_dir(template_root, dataset_type)
        pack_dataset(dataset_dir, template_dir)

if __name__ == "__main__":
    args = parse_args()
    print("is_dataset:", is_dataset(args.src_dir))
    print("is_collection_of_datasets:", is_collection_of_datasets(args.src_dir))