import shutil
from logging import WARN
from pathlib import Path

import pytest
from helpers import assert_git_status, assert_log, setup_git_repo

from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types.datasets import (
    Dataset,
    dataset_has_sample_type,
    save_datasets,
    save_datasets_auto_format,
)
from ada_eval.datasets.types.samples import (
    AdaSample,
    EvaluatedAdaSample,
    EvaluatedExplainSample,
    EvaluatedSample,
    EvaluatedSparkSample,
    ExplainSample,
    GeneratedAdaSample,
    GeneratedExplainSample,
    GeneratedSample,
    GeneratedSparkSample,
    Sample,
    SampleKind,
    SampleStage,
    SparkSample,
)


def test_dataset_types():
    base_ada_dataset = Dataset(name="test_0", sample_type=AdaSample, samples=[])
    assert base_ada_dataset.kind is SampleKind.ADA
    assert base_ada_dataset.stage is SampleStage.INITIAL
    assert base_ada_dataset.dirname == "ada_test_0"
    assert dataset_has_sample_type(base_ada_dataset, Sample)
    assert dataset_has_sample_type(base_ada_dataset, AdaSample)
    assert not dataset_has_sample_type(base_ada_dataset, GeneratedSample)
    assert not dataset_has_sample_type(base_ada_dataset, GeneratedAdaSample)
    assert not dataset_has_sample_type(base_ada_dataset, EvaluatedSample)
    assert not dataset_has_sample_type(base_ada_dataset, EvaluatedAdaSample)
    assert not dataset_has_sample_type(base_ada_dataset, ExplainSample)
    assert not dataset_has_sample_type(base_ada_dataset, SparkSample)

    generated_explain_dataset = Dataset(
        name="test_1", sample_type=GeneratedExplainSample, samples=[]
    )
    assert generated_explain_dataset.kind is SampleKind.EXPLAIN
    assert generated_explain_dataset.stage is SampleStage.GENERATED
    assert generated_explain_dataset.dirname == "explain_test_1"
    assert dataset_has_sample_type(generated_explain_dataset, Sample)
    assert dataset_has_sample_type(generated_explain_dataset, ExplainSample)
    assert dataset_has_sample_type(generated_explain_dataset, GeneratedSample)
    assert dataset_has_sample_type(generated_explain_dataset, GeneratedExplainSample)
    assert not dataset_has_sample_type(generated_explain_dataset, EvaluatedSample)
    assert not dataset_has_sample_type(
        generated_explain_dataset, EvaluatedExplainSample
    )
    assert not dataset_has_sample_type(generated_explain_dataset, AdaSample)
    assert not dataset_has_sample_type(generated_explain_dataset, SparkSample)

    evaluated_spark_dataset = Dataset(
        name="test_2", sample_type=EvaluatedSparkSample, samples=[]
    )
    assert evaluated_spark_dataset.kind is SampleKind.SPARK
    assert evaluated_spark_dataset.stage is SampleStage.EVALUATED
    assert evaluated_spark_dataset.dirname == "spark_test_2"
    assert dataset_has_sample_type(evaluated_spark_dataset, Sample)
    assert dataset_has_sample_type(evaluated_spark_dataset, AdaSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, SparkSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, GeneratedSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, GeneratedAdaSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, GeneratedSparkSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, EvaluatedSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, EvaluatedAdaSample)
    assert dataset_has_sample_type(evaluated_spark_dataset, EvaluatedSparkSample)
    assert not dataset_has_sample_type(evaluated_spark_dataset, ExplainSample)

    assert dataset_has_sample_type(
        generated_explain_dataset, (GeneratedSample, AdaSample, EvaluatedSparkSample)
    )
    assert not dataset_has_sample_type(
        generated_explain_dataset, (AdaSample, EvaluatedSparkSample)
    )


def test_save_datasets_packed(tmp_path: Path, generated_test_datasets: Path):
    # Initialise a Git repository to track changes
    setup_git_repo(tmp_path, initial_commit=True)
    assert_git_status(tmp_path, expect_dirty=False)

    # Load the dataset files
    datasets = load_datasets(generated_test_datasets)

    # Delete the dataset files
    for file in generated_test_datasets.iterdir():
        file.unlink()
    assert len(list(generated_test_datasets.iterdir())) == 0
    assert_git_status(tmp_path, expect_dirty=True)

    # `save_datasets()` should overwrite anything present, so add a file
    # which we expect to be removed
    test_file = generated_test_datasets / "test_file.txt"
    test_file.write_text("This file should be removed.")
    assert test_file.exists()

    # Save the datasets with `save_datasets()`
    save_datasets(datasets, generated_test_datasets)

    # This should have regenerated the original dataset files and removed
    # `test_file`
    assert len(list(generated_test_datasets.iterdir())) > 0
    assert_git_status(tmp_path, expect_dirty=False)
    assert not test_file.exists()

    # This should also work if the directory does not initially exist
    shutil.rmtree(generated_test_datasets)
    assert not generated_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets(datasets, generated_test_datasets)
    assert generated_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=False)


def test_save_datasets_unpacked(tmp_path: Path, expanded_test_datasets: Path):
    # Initialise a Git repository to track changes
    setup_git_repo(tmp_path, initial_commit=True)
    assert_git_status(tmp_path, expect_dirty=False)

    # Load the dataset files
    datasets = load_datasets(expanded_test_datasets)

    # Delete the dataset files
    for path in expanded_test_datasets.iterdir():
        shutil.rmtree(path)
    assert len(list(expanded_test_datasets.iterdir())) == 0
    assert_git_status(tmp_path, expect_dirty=True)

    # `save_datasets()` should overwrite anything present, so add a file
    # which we expect to be removed
    test_file = expanded_test_datasets / "test_file.txt"
    test_file.write_text("This file should be removed.")
    assert test_file.exists()

    # Save the datasets with `save_datasets()`
    save_datasets(datasets, expanded_test_datasets, unpacked=True)

    # This should have regenerated the original dataset files and removed
    # `test_file`.
    assert len(list(expanded_test_datasets.iterdir())) > 0
    assert not test_file.exists()

    # There will be git changes, as we did not commit a `comments.md` or
    # `prompt.md` file for `spark_test/test_sample_2`, but these will have been
    # populated with default values, and therefore created (empty) during saving.
    assert_git_status(tmp_path, expect_dirty=True)
    spark_sample_2_dir = expanded_test_datasets / "spark_test" / "test_sample_2"
    assert (spark_sample_2_dir / "comments.md").read_text() == ""
    assert (spark_sample_2_dir / "prompt.md").read_text() == ""

    # Remove the offending files and check that the git status is clean to
    # verify that the files were regenerated correctly
    (spark_sample_2_dir / "comments.md").unlink()
    (spark_sample_2_dir / "prompt.md").unlink()
    assert_git_status(tmp_path, expect_dirty=False)

    # This should also work if the directory does not initially exist
    shutil.rmtree(expanded_test_datasets)
    assert not expanded_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets(datasets, expanded_test_datasets, unpacked=True)
    assert expanded_test_datasets.exists()
    (spark_sample_2_dir / "comments.md").unlink()
    (spark_sample_2_dir / "prompt.md").unlink()
    assert_git_status(tmp_path, expect_dirty=False)


def test_save_datasets_auto_format(
    tmp_path: Path,
    expanded_test_datasets: Path,
    compacted_test_datasets: Path,
    caplog: pytest.LogCaptureFixture,
):
    # Touch the `comments.md` and `prompt.md` files in spark sample 2 so that
    # saving over the unpacked datasets restores the original state
    spark_sample_2_dir = expanded_test_datasets / "spark_test" / "test_sample_2"
    (spark_sample_2_dir / "comments.md").touch()
    (spark_sample_2_dir / "prompt.md").touch()

    # Initialise a Git repository to track changes
    setup_git_repo(tmp_path, initial_commit=True)
    assert_git_status(tmp_path, expect_dirty=False)

    # Load the dataset files
    datasets = load_datasets(expanded_test_datasets)
    datasets_dict = {d.dirname: d for d in datasets}

    # Test that packed format is used by default when there is no existing data
    save_datasets_auto_format(datasets, tmp_path / "new")
    assert_git_status(tmp_path, expect_dirty=True)
    assert (tmp_path / "new" / "ada_test.jsonl").exists()
    assert not (tmp_path / "new" / "ada_test").exists()
    shutil.rmtree(tmp_path / "new")
    assert_git_status(tmp_path, expect_dirty=False)

    # Test saving over a directory of unpacked datasets
    shutil.rmtree(expanded_test_datasets / "ada_test")
    shutil.rmtree(expanded_test_datasets / "explain_test")
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format(datasets, expanded_test_datasets)
    assert_git_status(tmp_path, expect_dirty=False)

    # Test saving over a directory of packed datasets
    (compacted_test_datasets / "ada_test.jsonl").unlink()
    (compacted_test_datasets / "explain_test.jsonl").unlink()
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format(datasets, compacted_test_datasets)
    assert_git_status(tmp_path, expect_dirty=False)

    # Test saving over a directory containing a mixture of packed and unpacked
    assert caplog.records == []
    shutil.copytree(
        expanded_test_datasets / "ada_test", compacted_test_datasets / "ada_test"
    )
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format(datasets, compacted_test_datasets)
    assert_git_status(tmp_path, expect_dirty=False)
    warn_msg = (
        f"Output path '{compacted_test_datasets}' contains a mixture of packed "
        "and unpacked data; Defaulting to packed format."
    )
    assert_log(caplog, WARN, warn_msg)
    caplog.clear()

    # Test saving over a collection of unpacked datasets which also contains
    # samples (or equivalently, an unpacked dataset which also contains other
    # datasets).
    for sample_dir in (expanded_test_datasets / "spark_test").iterdir():
        shutil.copytree(sample_dir, expanded_test_datasets / sample_dir.name)
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format(datasets, expanded_test_datasets)
    assert_git_status(tmp_path, expect_dirty=False)
    warn_msg = (
        f"Output path '{expanded_test_datasets}' contains a mixture of datasets "
        "and samples."
    )
    assert_log(caplog, WARN, warn_msg)
    caplog.clear()

    # Test saving a single dataset to a matching packed dataset file
    explain_dataset = datasets_dict["explain_test"]
    packed_explain_path = compacted_test_datasets / "explain_test.jsonl"
    packed_explain_path.write_text("")
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format([explain_dataset], packed_explain_path)
    assert_git_status(tmp_path, expect_dirty=False)

    # Test saving a single dataset to a matching unpacked dataset directory
    spark_dataset = datasets_dict["spark_test"]
    shutil.rmtree(expanded_test_datasets / "spark_test" / "test_sample_1")
    assert_git_status(tmp_path, expect_dirty=True)
    save_datasets_auto_format([spark_dataset], expanded_test_datasets / "spark_test")
    assert_git_status(tmp_path, expect_dirty=False)

    # Test saving a single dataset to a non-matching packed dataset file
    matching_path = compacted_test_datasets / "explain_test.jsonl"
    non_matching_path = compacted_test_datasets / "explain_other.jsonl"
    shutil.copy(matching_path, non_matching_path)
    assert non_matching_path.is_file()
    save_datasets_auto_format([explain_dataset], non_matching_path)
    assert non_matching_path.is_dir()
    saved_file = non_matching_path / "explain_test.jsonl"
    assert saved_file.is_file()
    assert saved_file.read_text() == matching_path.read_text()

    # Test saving a single dataset to a non-matching unpacked dataset directory
    matching_path = expanded_test_datasets / "spark_test"
    non_matching_path = expanded_test_datasets / "spark_other"
    shutil.copytree(matching_path, non_matching_path)
    for i in range(3):
        assert (non_matching_path / f"test_sample_{i}" / "other.json").is_file()
    save_datasets_auto_format([spark_dataset], non_matching_path)
    saved_dir = non_matching_path / "spark_test"
    for i in range(3):
        assert not (non_matching_path / f"test_sample_{i}").exists()
        assert (saved_dir / f"test_sample_{i}" / "other.json").is_file()

    # No unexpected log messages should have been emitted
    assert caplog.records == []
