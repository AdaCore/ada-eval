import json
from pathlib import Path

from ada_eval.datasets.pack_unpack import unpack_datasets
from ada_eval.datasets.types import (
    EVALUATION_RESULTS_KEY,
    GENERATED_SOLUTION_DIR_NAME,
    GENERATION_STATS_KEY,
)
from ada_eval.datasets.types.samples import OTHER_JSON_NAME


def test_unpack_generated_dataset(generated_test_datasets: Path, tmp_path: Path):
    """Test unpacking a generated dataset includes expected extra info."""
    generated_dataset_path = generated_test_datasets / "spark_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=generated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "spark_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    assert len(sample_dirs) > 0

    sample_dir = sample_dirs[0]

    # Check that the unpacked dataset includes a non-empty generated_solution dir
    generated_solution_dir = sample_dir / GENERATED_SOLUTION_DIR_NAME
    assert generated_solution_dir.is_dir()
    assert list(generated_solution_dir.iterdir())

    # Check that non-empty generation_stats are included in other.json
    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        assert GENERATION_STATS_KEY in other_data
        assert other_data[GENERATION_STATS_KEY]


def test_unpack_evaluated_dataset(evaluated_test_datasets: Path, tmp_path: Path):
    """Test unpacking an evaluated dataset includes expected extra info."""
    evaluated_dataset_path = evaluated_test_datasets / "spark_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=evaluated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "spark_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    sample_dir = sample_dirs[0]

    # Check that the unpacked dataset includes a non-empty generated_solution dir
    generated_solution_dir = sample_dir / GENERATED_SOLUTION_DIR_NAME
    assert (generated_solution_dir).is_dir()
    assert list(generated_solution_dir.iterdir())

    # Check that non-empty evaluation_results are included in other.json
    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        assert EVALUATION_RESULTS_KEY in other_data
        assert other_data[EVALUATION_RESULTS_KEY]
