import json
from pathlib import Path

from ada_eval.datasets.pack_unpack import unpack_datasets
from ada_eval.datasets.types import (
    EVALUATION_RESULTS_KEY,
    GENERATED_SOLUTION_DIR_NAME,
    GENERATED_SOLUTION_KEY,
    GENERATION_STATS_KEY,
)
from ada_eval.datasets.types.samples import OTHER_JSON_NAME


def test_unpack_generated_spark_dataset(generated_test_datasets: Path, tmp_path: Path):
    """Test unpacking a generated dataset includes expected extra info."""
    generated_dataset_path = generated_test_datasets / "spark_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=generated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "spark_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    assert len(sample_dirs) == 3

    sample_dir = sample_dirs[0]

    # Check that the unpacked dataset includes a non-empty generated_solution dir
    generated_solution_dir = sample_dir / GENERATED_SOLUTION_DIR_NAME
    assert generated_solution_dir.is_dir()
    generated_file = generated_solution_dir / "generated_file"
    assert generated_file.read_text() == "This file was added during generation\n"

    # Check that non-empty generation_stats are included in other.json
    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        assert GENERATION_STATS_KEY in other_data
        assert other_data[GENERATION_STATS_KEY] == {
            "exit_code": 0,
            "stdout": "This is the generation's stdout\n",
            "stderr": "",
            "runtime_ms": 0,
        }


def test_unpack_evaluated_spark_dataset(evaluated_test_datasets: Path, tmp_path: Path):
    """Test unpacking an evaluated dataset includes expected extra info."""
    evaluated_dataset_path = evaluated_test_datasets / "spark_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=evaluated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "spark_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    assert len(sample_dirs) == 3
    sample_dir = sample_dirs[0]

    # Check that the unpacked dataset includes a non-empty generated_solution dir
    generated_solution_dir = sample_dir / GENERATED_SOLUTION_DIR_NAME
    assert (generated_solution_dir).is_dir()
    generated_file = generated_solution_dir / "generated_file"
    assert generated_file.read_text() == "This file was added during generation\n"

    # Check that non-empty evaluation_results are included in other.json
    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        assert EVALUATION_RESULTS_KEY in other_data
        eval_results = other_data[EVALUATION_RESULTS_KEY]
        assert {e["eval"] for e in eval_results} == {"build", "prove"}


def test_unpack_generated_explain_dataset(
    generated_test_datasets: Path, tmp_path: Path
):
    """Test unpacking a generated dataset includes expected extra info."""
    generated_dataset_path = generated_test_datasets / "explain_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=generated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "explain_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    assert len(sample_dirs) == 1

    sample_dir = sample_dirs[0]

    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        # Check that the generated solution is included in other.json
        assert GENERATED_SOLUTION_KEY in other_data
        assert (
            other_data[GENERATED_SOLUTION_KEY] == "This is the generated explanation."
        )
        # Check that the generation_stats are included in other.json
        assert GENERATION_STATS_KEY in other_data
        assert other_data[GENERATION_STATS_KEY] == {
            "exit_code": 0,
            "stdout": "This is the generation's stdout\n",
            "stderr": "",
            "runtime_ms": 0,
        }


def test_unpack_evaluated_explain_dataset(
    evaluated_test_datasets: Path, tmp_path: Path
):
    """Test unpacking an evaluated dataset includes expected extra info."""
    evaluated_dataset_path = evaluated_test_datasets / "explain_test.jsonl"
    expanded_dir = tmp_path / "expanded"
    unpack_datasets(src=evaluated_dataset_path, dest_dir=expanded_dir)

    unpacked_dataset_dir = expanded_dir / "explain_test"
    sample_dirs = sorted([d for d in unpacked_dataset_dir.iterdir() if d.is_dir()])
    assert len(sample_dirs) == 1
    sample_dir = sample_dirs[0]

    with (sample_dir / OTHER_JSON_NAME).open() as f:
        other_data = json.load(f)
        # Check that the generated solution is included in other.json
        assert GENERATED_SOLUTION_KEY in other_data
        assert (
            other_data[GENERATED_SOLUTION_KEY] == "This is the generated explanation."
        )
        # Check that the evaluation_results are included in other.json
        assert EVALUATION_RESULTS_KEY in other_data
        eval_results = other_data[EVALUATION_RESULTS_KEY]
        # I know these stats don't make sense for explain, but we don't specific
        # explain metrics setup yet, and this at least checks the unpacking works.
        assert {e["eval"] for e in eval_results} == {"build", "prove"}
