"""Script to pack one or more datasets from the data folder. This takes one or
more projects and packs them into a single jsonl file.

This is the reverse process of the unpack_datasets.py script.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# from ada_eval.common_types import DatasetType
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR, DATASET_TEMPLATES_DIR
from ada_eval.common_types import DatasetType, SampleTemplate, AdaSample, ExplainSolution, ExplainSample, SparkSample, Location
import subprocess

# Unpacked samples, will always have one file and two dirs: "base", "solution", and "other.json"
BASE_DIR_NAME = "base"
SOLUTION_DIR_NAME = "solution"
UNIT_TEST_DIR_NAME = "unit_test"
OTHER_JSON_NAME = "other.json"
COMMENTS_FILE = "comments.md"
PROMPT_FILE = "prompt.md"

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

def make_files_relative_to(path: Path, files: list[Path]) -> list[Path]:
    """Makes a list of files relative to a given path"""
    return [file.relative_to(path) for file in files]

def get_sample_prompt(sample_root: Path) -> str:
    """Returns the prompt for a sample"""
    prompt_file = sample_root / PROMPT_FILE
    if not prompt_file.is_file():
        return ""
    return prompt_file.read_text(encoding="utf-8")

def get_sample_comments(sample_root: Path) -> str:
    """Returns any comments for a sample"""
    comments_file = sample_root / COMMENTS_FILE
    if not comments_file.is_file():
        return ""
    return comments_file.read_text(encoding="utf-8")

def filter_template_files(files: list[tuple[Path, Path]], dataset: UnpackedDataSetMetadata) -> list[Path]:
    """Removes template files from the list of files.

    Args:
        files (list[tuple[Path, Path]]): List of file paths. Each tuple contains the short path (relative to the sample's base dir) and the full path.
        dataset (UnpackedDataSetMetadata): metadata for the dataset that contains the sample

    Returns:
        list[Path]: list of filtered full paths
    """
    res = []
    for short, long in files:
        if short not in dataset.sample_template.sources:
            res.append(long)
        elif long.read_text() != dataset.sample_template.sources[short]:
            res.append(long)
    return res

def git_ls_files(root: Path) -> list[Path]:
    """Returns a list of files in a directory using git ls-files"""
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        check=True
    )

    git_files = [root / line for line in result.stdout.splitlines()]

    # We have to check that a path exists, as git ls-files will return files that
    # were previously committed but have since been deleted.
    return [path for path in git_files if path.is_file()]

def get_non_template_files(root: Path, dataset: UnpackedDataSetMetadata) -> list[Path]:
    """Returns a list of files in a directory that are not in the template"""
    full_paths = git_ls_files(root)
    short_paths = make_files_relative_to(root, full_paths)
    unique_files = filter_template_files(zip(short_paths, full_paths), dataset)
    return unique_files

def pack_dataset(dataset: UnpackedDataSetMetadata, dest_dir: Path) -> list[AdaSample] | list[ExplainSample] | list[SparkSample]:
    """Packs a dataset into a jsonl file"""
    dest_file = dest_dir / f"{dataset.type.value}.jsonl"
    dest_file.write_text("")
    for sample_dir in dataset.dir.iterdir():
        if not is_sample(sample_dir):
            continue
        other_json_file = sample_dir / "other.json"
        other_data = dataset.sample_template.others | json.loads(other_json_file.read_text())

        # Use git ls-files to get the list of all files
        unique_base_files = get_non_template_files(sample_dir / BASE_DIR_NAME, dataset)
        unique_solution_files = get_non_template_files(sample_dir /SOLUTION_DIR_NAME, dataset)
        unique_unit_test_files = get_non_template_files(sample_dir / UNIT_TEST_DIR_NAME, dataset)

        prompt = get_sample_prompt(sample_dir)
        comments = get_sample_comments(sample_dir)

        match dataset.type:
            case DatasetType.ADA | DatasetType.SPARK:
                sample_class = AdaSample if dataset.type == DatasetType.ADA else SparkSample
                sample = sample_class(
                    name=sample_dir.name,
                    location=Location.from_dict(other_data["location"]),
                    prompt=prompt,
                    comments=comments,
                    sources={p: p.read_text(encoding="utf-8") for p in unique_base_files},
                    canonical_solution={p: p.read_text(encoding="utf-8") for p in unique_solution_files},
                    unit_tests={p: p.read_text(encoding="utf-8") for p in unique_unit_test_files},
                )
            case DatasetType.EXPLAIN:
                sample = ExplainSample(
                    name=sample_dir.name,
                    location=Location.from_dict(other_data["location"]),
                    prompt=prompt,
                    comments=comments,
                    sources={p: p.read_text(encoding="utf-8") for p in unique_base_files},
                    canonical_solution={p: p.read_text(encoding="utf-8") for p in unique_solution_files},
                    unit_tests={p: p.read_text(encoding="utf-8") for p in unique_unit_test_files},
                )
            case _:
                raise ValueError(f"Unknown dataset type: {dataset.type}")
        # Write the sample to the jsonl file
        with open(dest_file, "a") as f:
            f.write(sample.to_json() + "\n")

def pack_datasets(datasets: list[UnpackedDataSetMetadata], dest_dir: Path):
    """Packs each datasets into into a jsonl file"""
    for dataset in datasets:
        pack_dataset(dataset, dest_dir)

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
    pack_datasets(datasets, args.dest_dir)
