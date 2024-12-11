from pathlib import Path
import shutil

from ada_eval.datasets.loader import load_packed_dataset
from ada_eval.datasets.types.datasets import Dataset
from ada_eval.datasets.utils import get_packed_dataset_files
from ada_eval.paths import GENERATION_WORKING_DIR

def unpack_dataset_for_generation(dataset: Dataset):
    # Remove files from previous runs
    dataset_working_dir = GENERATION_WORKING_DIR / f"{dataset.type}_{dataset.name}"
    shutil.rmtree(dataset_working_dir, ignore_errors=True)

    # Don't ignore errors so we know if it fails to clean up previous files
    dataset_working_dir.mkdir()

    # Unpack each sample
    for sample in dataset.samples:
        sample_working_dir = dataset_working_dir / sample.name
        sample_working_dir.mkdir()
        sample.unpack_for_generation(sample_working_dir)

def generate_completions(packed_dataset_or_dir: Path):
    dataset_files = get_packed_dataset_files(packed_dataset_or_dir)
    GENERATION_WORKING_DIR.mkdir(exist_ok=True)
    for path in dataset_files:
        dataset = load_packed_dataset(path)
        unpack_dataset_for_generation(dataset)
