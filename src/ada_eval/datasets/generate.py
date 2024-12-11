from pathlib import Path

# from ada_eval.datasets.loader import load_packed_dataset
from ada_eval.datasets.utils import get_packed_dataset_files


def generate_completions(packed_dataset_or_dir: Path):
    dataset_files = get_packed_dataset_files(packed_dataset_or_dir)
    for path in dataset_files:
        pass
        # dataset = load_packed_dataset(path)
        # dataset.generate_completions()
