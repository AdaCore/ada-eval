from pathlib import Path

from ada_eval.datasets.loader import load_packed_dataset, load_unpacked_dataset
from ada_eval.datasets.utils import get_packed_dataset_files, get_unpacked_dataset_dirs


def unpack_datasets(src: Path, dest_dir: Path, force: bool = False):
    # TODO add docstring
    # TODO only overwrite uncommitted changes if force is set
    dataset_files = get_packed_dataset_files(src)
    for path in dataset_files:
        dataset = load_packed_dataset(path)
        dataset.save_unpacked(dest_dir)


def pack_datasets(src_dir: Path, dest_dir: Path, force: bool = False):
    # TODO add docstring
    # TODO only overwrite uncommitted changes if force is set
    dataset_dirs = get_unpacked_dataset_dirs(src_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)
    for path in dataset_dirs:
        dataset = load_unpacked_dataset(path)
        dataset.save_packed(dest_dir)
