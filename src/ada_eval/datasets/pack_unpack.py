import sys
from pathlib import Path

from ada_eval.datasets.utils import is_git_up_to_date

from .loader import load_packed_dataset, load_unpacked_dataset
from .types import get_packed_dataset_files, get_unpacked_dataset_dirs


def unpack_datasets(src: Path, dest_dir: Path, *, force: bool = False):
    if dest_dir.exists() and not is_git_up_to_date(dest_dir) and not force:
        print(
            f"Changes to dataset files in {dest_dir} have not been committed. "
            "Either commit them or re-run with --force"
        )
        sys.exit(1)
    dataset_files = get_packed_dataset_files(src)
    for path in dataset_files:
        dataset = load_packed_dataset(path)
        dataset.save_unpacked(dest_dir)


def pack_datasets(src_dir: Path, dest_dir: Path, *, force: bool = False):
    # TODO improve this check to only consider the file of interest,
    # instead of every file in the destination dir
    if dest_dir.exists() and not is_git_up_to_date(dest_dir) and not force:
        print(
            f"Changes to dataset files in {dest_dir} have not been committed. "
            "Either commit them or re-run with --force"
        )
        sys.exit(1)
    dataset_dirs = get_unpacked_dataset_dirs(src_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)
    for path in dataset_dirs:
        dataset = load_unpacked_dataset(path)
        dataset.save_packed(dest_dir)
