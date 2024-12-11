from pathlib import Path

from ada_eval.datasets.loader import load_packed_dataset, load_unpacked_dataset
from ada_eval.datasets.utils import get_unpacked_dataset_dirs, get_packed_dataset_files


def unpack_datasets(src: Path, dest_dir: Path, force: bool = False):
    dataset_files = get_packed_dataset_files(src)
    for path in dataset_files:
        dataset = load_packed_dataset(path)
        dataset.save_unpacked(dest_dir)


def pack_datasets(src_dir: Path, dest_dir: Path):
    dataset_dirs = get_unpacked_dataset_dirs(src_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)
    for path in dataset_dirs:
        dataset = load_unpacked_dataset(path)
        dataset.save_packed(dest_dir)
