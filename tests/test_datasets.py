import shutil
import subprocess
from pathlib import Path

from helpers import (
    assert_git_status,
    generated_test_datasets,  # noqa: F401  # Fixtures used implicitly
    setup_git_repo,
)

from ada_eval.datasets.loader import load_dir
from ada_eval.datasets.types.datasets import (
    Dataset,
    DatasetKind,
    dataset_has_sample_type,
    save_to_dir_packed,
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
    SparkSample,
)


def test_dataset_kind_str():
    assert str(DatasetKind.ADA) == "ada"
    assert str(DatasetKind.EXPLAIN) == "explain"
    assert str(DatasetKind.SPARK) == "spark"


def test_dataset_types():
    base_ada_dataset = Dataset(name="test_0", sample_type=AdaSample, samples=[])
    assert base_ada_dataset.kind() is DatasetKind.ADA
    assert base_ada_dataset.dirname() == "ada_test_0"
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
    assert generated_explain_dataset.kind() is DatasetKind.EXPLAIN
    assert generated_explain_dataset.dirname() == "explain_test_1"
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
    assert evaluated_spark_dataset.kind() is DatasetKind.SPARK
    assert evaluated_spark_dataset.dirname() == "spark_test_2"
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


def test_save_to_dir_packed(tmp_path: Path, generated_test_datasets: Path):  # noqa: F811  # pytest fixture
    # Initialise a Git repository to track changes
    setup_git_repo(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "msg"], cwd=tmp_path, check=True)
    assert_git_status(tmp_path, expect_dirty=False)

    # Load the dataset files
    datasets = load_dir(generated_test_datasets)

    # Delete the dataset files
    for file in generated_test_datasets.iterdir():
        file.unlink()
    assert len(list(generated_test_datasets.iterdir())) == 0
    assert_git_status(tmp_path, expect_dirty=True)

    # `save_to_dir_packed()` should overwrite anything present, so add a file
    # which we expect to be removed
    test_file = generated_test_datasets / "test_file.txt"
    test_file.write_text("This file should be removed.")
    assert test_file.exists()

    # Save the datasets with `save_to_dir_packed()`
    save_to_dir_packed(datasets, generated_test_datasets)

    # This should have regenerated the original dataset files and removed
    # `test_file`
    assert len(list(generated_test_datasets.iterdir())) > 0
    assert_git_status(tmp_path, expect_dirty=False)
    assert not test_file.exists()

    # This should also work if the directory does not initially exist
    shutil.rmtree(generated_test_datasets)
    assert not generated_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=True)
    save_to_dir_packed(datasets, generated_test_datasets)
    assert generated_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=False)
