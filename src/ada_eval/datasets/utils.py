import subprocess
from pathlib import Path

from ada_eval.datasets.types import OTHER_JSON_NAME


def is_packed_dataset(path: Path) -> bool:
    """Returns true if this is the path to a packed dataset. This is the case if
    it's a path to a jsonl file."""
    return path.is_file() and path.suffix == ".jsonl"


def is_collection_of_packed_datasets(path: Path) -> bool:
    """Returns true if this is the path to a collection of packed datasets. This
    is the case if the dir contains a jsonl file."""
    return path.is_dir() and any(is_packed_dataset(p) for p in path.iterdir())


def is_unpacked_sample(path: Path) -> bool:
    """Returns true if this is the path to a sample. This is the case if the dir
    contains an OTHER_JSON_NAME file."""
    other_json = path / OTHER_JSON_NAME
    return path.is_dir() and other_json.is_file()


def is_unpacked_dataset(path: Path) -> bool:
    """Returns true if this is a path to a directory that contains an unpacked
    dataset. For this to be true, at least one child dir must contain a sample."""
    return path.is_dir() and any(is_unpacked_sample(d) for d in path.iterdir())


def is_collection_of_unpacked_datasets(path: Path) -> bool:
    """Return's true if this is a path to a directory that contains multiple datasets.
    Note that not all directories in this directory need to contain a dataset."""
    return path.is_dir() and any(is_unpacked_dataset(d) for d in path.iterdir())


def is_git_up_to_date(path: Path) -> bool:
    """Returns true if the contents of a folder are up to date in git. In this
    context we mean that no changes have been made to the files in the folder,
    including file creations/deletions/modifications."""
    result = subprocess.run(
        ["git", "status", "--porcelain=1", "."],
        encoding="utf-8",
        capture_output=True,
        cwd=path,
    )
    return result.returncode == 0 and (
        result.stdout is None or result.stdout.strip() == ""
    )


def get_packed_dataset_files(path: Path) -> list[Path]:
    """Returns a list of paths to the files in the dataset."""
    if is_packed_dataset(path):
        return [path]
    if is_collection_of_packed_datasets(path):
        return [p for p in path.iterdir() if is_packed_dataset(p)]
    return []


def get_unpacked_dataset_dirs(path: Path) -> list[Path]:
    """Returns a list of paths that contain the unpacked contents of a dataset."""
    if not is_collection_of_unpacked_datasets(path) and not is_unpacked_dataset(path):
        return []
    if is_unpacked_dataset(path):
        return [path]
    return [x for x in path.iterdir() if is_unpacked_dataset(x)]
