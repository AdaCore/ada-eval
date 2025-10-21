import re
import shutil
from pathlib import Path

import pytest
from helpers import check_test_datasets  # noqa: F401  # Fixtures used implicitly

from ada_eval.check_datasets import check_base_datasets
from ada_eval.datasets import Dataset, dataset_has_sample_type, load_datasets
from ada_eval.datasets.types.datasets import DatasetsMismatchError, save_datasets
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.samples import (
    GeneratedSparkSample,
    GenerationStats,
    SparkSample,
)


def test_check_base_datasets(tmp_path: Path, check_test_datasets: Path):  # noqa: F811  # pytest fixture
    # Load a dataset which passes all checks.
    correct_dataset = load_datasets(check_test_datasets / "spark_check.jsonl")[0]
    assert dataset_has_sample_type(correct_dataset, SparkSample)
    good_sample = correct_dataset.samples[0]
    correct_dataset = Dataset(
        name="check", sample_type=SparkSample, samples=[good_sample]
    )

    # Save this dataset in both expanded and compacted forms
    expanded_dir = tmp_path / "expanded"
    save_datasets([correct_dataset], expanded_dir, unpacked=True)
    compacted_dir = tmp_path / "compacted"
    save_datasets([correct_dataset], compacted_dir, unpacked=False)

    # The datasets are correct and matching, so `check_base_datasets()` should
    # not raise any exception
    check_base_datasets(expanded_dir, compacted_dir)

    # Check that a missing dataset is detected
    shutil.copytree(expanded_dir / "spark_check", expanded_dir / "spark_other")
    error_msg = "dataset 'spark_other' is only present in the expanded datasets."
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        check_base_datasets(expanded_dir, compacted_dir)

    # Check that differing sample types are detected
    generated_sample = GeneratedSparkSample(
        **good_sample.model_dump(),
        generation_stats=GenerationStats(
            exit_code=0, stdout="", stderr="", runtime_ms=0
        ),
        generated_solution=DirectoryContents({}),
    )
    generated_dataset = Dataset(
        name="check", sample_type=GeneratedSparkSample, samples=[generated_sample]
    )
    save_datasets([correct_dataset], expanded_dir, unpacked=True)
    save_datasets([generated_dataset], compacted_dir, unpacked=False)
    error_msg = (
        "dataset 'spark_check' has type 'SparkSample' in the expanded datasets but "
        "type 'GeneratedSparkSample' in the compacted datasets."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        check_base_datasets(expanded_dir, compacted_dir)

    # Check that missing samples are detected
    two_sample_dataset = Dataset(
        name="check",
        sample_type=SparkSample,
        samples=[good_sample, good_sample.model_copy(update={"name": "name2"})],
    )
    save_datasets([two_sample_dataset], expanded_dir, unpacked=True)
    save_datasets([correct_dataset], compacted_dir, unpacked=False)
    error_msg = (
        "sample 'name2' of dataset 'spark_check' is only present in the "
        "expanded datasets."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        check_base_datasets(expanded_dir, compacted_dir)

    # Check that differing samples are detected
    modified_sample = good_sample.model_copy(deep=True)
    modified_sample.canonical_solution.files[Path("src/foo.adb")] = "compacted nested"
    modified_dataset = Dataset(
        name="check", sample_type=SparkSample, samples=[modified_sample]
    )
    save_datasets([modified_dataset], compacted_dir, unpacked=False)
    modified_sample.canonical_solution.files[Path("src/foo.adb")] = "expanded nested"
    modified_sample.sources.files[Path("new_file")] = "new"
    modified_sample.prompt = "Modified prompt"
    save_datasets([modified_dataset], expanded_dir, unpacked=True)
    error_msg = (
        "sample 'good' of dataset 'spark_check' differs between the "
        "expanded datasets and the compacted datasets:\n\n"
        "{'prompt': 'Modified prompt', 'sources': {PosixPath('new_file'): 'new'},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): 'expanded nested'}}"
        "\n\n{'prompt': '', 'sources': {},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): 'compacted nested'}}"
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        check_base_datasets(expanded_dir, compacted_dir)
