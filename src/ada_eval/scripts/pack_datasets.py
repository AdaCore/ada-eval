"""Script to pack one or more datasets from the data folder. This takes one or
more projects and packs them into a single jsonl file.

This is the reverse process of the unpack_datasets.py script.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

# from ada_eval.common_types import DatasetType
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR, DATASET_TEMPLATES_DIR
from ada_eval.common_types import DatasetType, SampleTemplate, AdaDataset, ExplainSolution, ExplainDataset, SparkDataset
import subprocess

# Unpacked samples, will always have one file and two dirs: "base", "solution", and "other.json"
BASE_DIR_NAME = "base"
SOLUTION_DIR_NAME = "solution"
OTHER_JSON_NAME = "other.json"

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
    arg_parser.add_argument("--src", type=Path, help="Source dir containing unpacked dataset or datasets", default=EXPANDED_DATASETS_DIR)
    arg_parser.add_argument("--dest", type=Path, help="Destination dir for packed dataset or datasets", default=COMPACTED_DATASETS_DIR)
    arg_parser.add_argument("--template_root", type=Path, help="Directory containing dataset templates", default=DATASET_TEMPLATES_DIR)
    args = arg_parser.parse_args()
    return Args(src_dir=args.src, dest_dir=args.dest, template_dir=args.template_root)

def is_sample(path: Path) -> bool:
    """Returns true if this is the path to a sample. This is the case if the dir
    contains an OTHER_JSON_NAME file."""
    other_json = path / OTHER_JSON_NAME
    return path.is_dir() and other_json.is_file()

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

def make_files_relative_to(path: Path, files: List[Path]) -> List[Path]:
    """Makes a list of files relative to a given path"""
    return [file.relative_to(path) for file in files]

def remove_template_files(files: List[Path], dataset: UnpackedDataSetMetadata) -> List[Path]:
    """Removes template files from the list of files"""
    res = []
    for file in files:
        if file.relative_to(dataset.dir / BASE_DIR_NAME) not in dataset.sample_template.sources:
            res.append(file)
        elif file.read_text() != dataset.sample_template.sources[file.relative_to(dataset.dir / BASE_DIR_NAME)]:
            res.append(file)
    return res

def git_ls_files(root: Path) -> List[Path]:
    """Returns a list of files in a directory using git ls-files"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        check=True
    )
    return [Path(line) for line in result.stdout.splitlines()]

def pack_base_dataset(dataset: UnpackedDataSetMetadata):
    pass

def pack_ada_dataset(dataset: UnpackedDataSetMetadata):
    pass

def pack_explain_dataset(dataset: UnpackedDataSetMetadata):
    pass

def pack_spark_dataset(dataset: UnpackedDataSetMetadata):
    pass

def pack_dataset(dataset: UnpackedDataSetMetadata):
    """Packs a dataset into a jsonl file"""
    for sample in dataset.dir.iterdir():
        print(sample)
        if not is_sample(sample):
            continue
        other_json_file = sample / "other.json"
        other_data = dataset.sample_template.others | json.loads(other_json_file.read_text())

        files = []
        # Use git ls-files to get the list of all files
        all_files = []



        for root, _, filenames in (sample / BASE_DIR_NAME).walk():
            for filename in filenames:
                files.append((root / filename).relative_to(sample))
        print("before:", len(files))
        files = remove_template_files(files, dataset)
        print("after:", len(files))

        match dataset.type:
            case DatasetType.ADA:
                pass
            case DatasetType.EXPLAIN:
                pass
            case DatasetType.SPARK:
                pass

def pack_datasets(datasets: list[UnpackedDataSetMetadata]):
    """Packs each datasets into into a jsonl file"""
    for dataset in datasets:
        print(dataset)
        pack_dataset(dataset)

def get_sample_template(template_dir: Path) -> SampleTemplate:
    """Returns the sample template for a dataset"""
    other_json_path = template_dir / OTHER_JSON_NAME
    other_json_contents = {}
    if other_json_path.is_file():
        other_json_contents = json.loads(other_json_path.read_text())
    files = {}
    for root, _, filenames in template_dir.walk():
        for file in filenames:
            file = root / file
            contents = file.read_text()
            file = file.relative_to(template_dir)
            files[file] = contents
    return SampleTemplate(sources=files, others=other_json_contents)

def get_datasets(path: Path, template_root: Path) -> list[UnpackedDataSetMetadata]:
    """Returns a list of datasets in the given path"""
    if not is_collection_of_datasets(path) and not is_dataset(path):
        return []

    if is_dataset(path):
        dataset_type = get_dataset_type(path)
        if dataset_type is None:
            return []
        template_dir = get_template_dir(template_root, dataset_type)
        sample_template = get_sample_template(template_dir)
        return [UnpackedDataSetMetadata(dir=path, type=dataset_type, sample_template=sample_template)]
    datasets = []
    for d in path.iterdir():
        if not is_dataset(d):
            continue
        dataset_type = get_dataset_type(d)
        if dataset_type is None:
            continue
        template_dir = get_template_dir(template_root, dataset_type)
        sample_template = get_sample_template(template_dir)
        datasets.append(UnpackedDataSetMetadata(dir=d, type=dataset_type, sample_template=sample_template))
    return datasets

if __name__ == "__main__":
    args = parse_args()
    datasets = get_datasets(args.src_dir, args.template_dir)
    print(args.src_dir)
    print(is_collection_of_datasets(args.src_dir))
    print(is_dataset(args.src_dir))
    pack_datasets(datasets)
