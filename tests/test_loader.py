import json
import re
import shutil
from logging import WARN
from pathlib import Path

import pydantic
import pytest
from helpers import (
    assert_log,
    compacted_test_datasets,  # noqa: F401  # Fixtures used implicitly
    evaluated_test_datasets,  # noqa: F401  # Fixtures used implicitly
    expanded_test_datasets,  # noqa: F401  # Fixtures used implicitly
    generated_test_datasets,  # noqa: F401  # Fixtures used implicitly
    setup_git_repo,
)

from ada_eval.datasets import Dataset, dataset_has_sample_type
from ada_eval.datasets.loader import (
    DuplicateSampleNameError,
    InvalidDatasetError,
    InvalidDatasetNameError,
    MixedDatasetFormatsError,
    MixedSampleTypesError,
    UnknownDatasetKindError,
    load_datasets,
    load_packed_dataset,
    load_unpacked_dataset,
)
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.evaluation_stats import (
    EvaluationStatsBuild,
    EvaluationStatsFailed,
    EvaluationStatsProve,
    EvaluationStatsTimedOut,
)
from ada_eval.datasets.types.samples import (
    EVALUATED_SAMPLE_TYPES,
    GENERATED_SAMPLE_TYPES,
    AdaSample,
    EvaluatedAdaSample,
    EvaluatedSample,
    ExplainSample,
    GeneratedSample,
    GenerationStats,
    Location,
    Sample,
    SampleKind,
    SampleStage,
    SparkSample,
)


def expected_base_sample_fields(
    sample_name: str, dataset_dirname: str
) -> dict[str, object]:
    """Return expected fields common to (almost) all `Samples` in the test datasets."""
    return {
        "name": sample_name,
        "location": Location(
            path=Path("source_file_0"), subprogram_name="My_Subprogram"
        ),
        "prompt": (
            f"This is the prompt for sample '{sample_name}' from dataset "
            f"'{dataset_dirname}'.\n"
        ),
        "sources": DirectoryContents(
            {
                Path("source_file_0"): (
                    f"This is 'source_file_0' in sample '{sample_name}' from "
                    f"dataset '{dataset_dirname}'.\n"
                )
            }
        ),
        "comments": (
            f"This is a comment on sample '{sample_name}' from dataset "
            f"'{dataset_dirname}'.\n"
        ),
        "canonical_evaluation_results": [],
    }


def expected_explain_sample(sample_name: str, dataset_dirname: str) -> ExplainSample:
    """Return an `ExplainSample` matching that expected from the test datasets."""
    return ExplainSample(
        **expected_base_sample_fields(sample_name, dataset_dirname),
        canonical_solution=(
            f"This is the reference answer for sample '{sample_name}' from "
            f"dataset '{dataset_dirname}'.\n"
        ),
        correct_statements=[
            "This is a correct statement.",
            "This is another correct statement.",
        ],
        incorrect_statements=[
            "This is an incorrect statement.",
            "This is another incorrect statement.",
        ],
    )


def expected_ada_sample(sample_name: str, dataset_dirname: str) -> AdaSample:
    """Return an `AdaSample` mostly matching those expected from the test datasets."""
    return AdaSample(
        **expected_base_sample_fields(sample_name, dataset_dirname),
        canonical_solution=DirectoryContents(
            {
                Path("source_file_0"): (
                    f"This is 'source_file_0' in sample '{sample_name}' from "
                    f"dataset '{dataset_dirname}'.\nThis is a new line added as "
                    "part of the canonical solution.\n"
                )
            }
        ),
        unit_tests=DirectoryContents(
            {
                Path("unit_test_file_0"): (
                    f"This is a unit test for sample '{sample_name}' from dataset "
                    f"'{dataset_dirname}'.\n"
                )
            }
        ),
    )


def expected_spark_sample(sample_name: str, dataset_dirname: str) -> SparkSample:
    """Return a `SparkSample` mostly matching those expected from the test datasets."""
    return SparkSample(
        **expected_ada_sample(sample_name, dataset_dirname).model_dump(),
    )


def expected_generated_sample(base_sample: Sample) -> GeneratedSample:
    """Return the expected `GeneratedSample` corresponding to a base sample."""
    if isinstance(base_sample, AdaSample):
        generated_solution: object = DirectoryContents(
            base_sample.sources.files
            | {Path("generated_file"): "This file was added during generation\n"}
        )
    else:
        generated_solution = "This is the generated explanation."
    return GENERATED_SAMPLE_TYPES[base_sample.kind](
        **base_sample.model_dump(),
        generation_stats=GenerationStats(
            exit_code=0,
            stdout="This is the generation's stdout\n",
            stderr="",
            runtime_ms=0,
        ),
        generated_solution=generated_solution,
    )


def expected_evaluated_sample(base_sample: Sample) -> EvaluatedSample:
    """Return the expected `EvaluatedSample` corresponding to a base sample."""
    generated_sample = expected_generated_sample(base_sample)
    return EVALUATED_SAMPLE_TYPES[generated_sample.kind](
        **generated_sample.model_dump(),
        evaluation_results=[
            EvaluationStatsProve(successfully_proven=False, subprogram_found=True),
            EvaluationStatsBuild(
                compiled=False, pre_format_warnings=True, post_format_warnings=True
            ),
        ],
    )


def check_loaded_datasets(
    datasets: list[Dataset[Sample]], stage: SampleStage = SampleStage.INITIAL
) -> None:
    """Check that `datasets` matches `tests/data/valid_[base/generated]_datasets`."""

    def promoted_if_needed(sample: Sample) -> Sample:
        match stage:
            case SampleStage.INITIAL:
                return sample
            case SampleStage.GENERATED:
                return expected_generated_sample(sample)
            case SampleStage.EVALUATED:
                return expected_evaluated_sample(sample)

    assert len(datasets) == 3
    datasets_by_name = {d.dirname: d for d in datasets}

    # Check the Explain dataset
    explain_dataset = datasets_by_name["explain_test"]
    assert explain_dataset.name == "test"
    assert explain_dataset.kind is SampleKind.EXPLAIN
    assert explain_dataset.stage is stage
    assert explain_dataset.samples == [
        promoted_if_needed(expected_explain_sample("test_sample_0", "explain_test"))
    ]

    # Construct the expected sample for the Ada dataset (i.e. that returned by
    # `expected_ada_sample()`, except with some `canonical_evaluation_results`)
    expected_ada_sample_0 = expected_ada_sample("test_sample_0", "ada_test")
    expected_ada_sample_0.canonical_evaluation_results = [
        EvaluationStatsBuild(
            compiled=True, pre_format_warnings=True, post_format_warnings=False
        )
    ]
    # Check the Ada dataset
    ada_dataset = datasets_by_name["ada_test"]
    assert ada_dataset.name == "test"
    assert ada_dataset.kind is SampleKind.ADA
    assert ada_dataset.stage is stage
    assert ada_dataset.samples == [promoted_if_needed(expected_ada_sample_0)]

    # Construct expected samples for the Spark dataset (sample_0 is mostly
    # as returned by `expected_spark_sample()`, sample_1 has some extra files,
    # and sample_2 is empty apart from a minimal `other.json` file, so should be
    # populated with defaults)
    expected_spark_sample_0 = expected_spark_sample("test_sample_0", "spark_test")
    expected_spark_sample_0.canonical_evaluation_results = [
        EvaluationStatsTimedOut(
            eval="prove", cmd_timed_out=["cmd", "arg0", "arg1"], timeout=12.34
        ),
        EvaluationStatsFailed(eval="build", exception='SomeError("Some message")'),
    ]
    expected_spark_sample_1 = expected_spark_sample("test_sample_1", "spark_test")
    expected_spark_sample_1.sources.files[Path("source_dir_0/source_file_1")] = (
        "This is 'source_file_1' in sample 'test_sample_1' from dataset 'spark_test'.\n"
    )
    expected_spark_sample_1.canonical_solution.files[
        Path("source_dir_1/source_file_2")
    ] = (
        "This is 'source_file_2' in sample 'test_sample_1' from dataset 'spark_test'.\n"
        "The addition of this file is part of the canonical solution.\n"
    )
    expected_spark_sample_1.canonical_evaluation_results = [
        EvaluationStatsProve(successfully_proven=True, subprogram_found=True),
        EvaluationStatsBuild(
            compiled=True, pre_format_warnings=False, post_format_warnings=False
        ),
    ]
    expected_spark_sample_2 = SparkSample(
        name="test_sample_2",
        location=Location(
            path=Path("non/existent/path"), subprogram_name="My_Subprogram"
        ),
        prompt="",
        sources=DirectoryContents({}),
        canonical_solution=DirectoryContents({}),
        canonical_evaluation_results=[],
        comments="",
        unit_tests=DirectoryContents({}),
    )
    # Check the Spark dataset (note that the sample ordering is not guaranteed)
    spark_dataset = datasets_by_name["spark_test"]
    assert spark_dataset.name == "test"
    assert spark_dataset.kind is SampleKind.SPARK
    assert spark_dataset.stage is stage
    spark_samples_by_name = {s.name: s for s in spark_dataset.samples}
    assert spark_samples_by_name == {
        "test_sample_0": promoted_if_needed(expected_spark_sample_0),
        "test_sample_1": promoted_if_needed(expected_spark_sample_1),
        "test_sample_2": promoted_if_needed(expected_spark_sample_2),
    }


def test_load_valid_unpacked_datasets(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that loading unpacked datasets works correctly."""
    check_loaded_datasets(load_datasets(expanded_test_datasets))


def test_load_valid_packed_datasets(compacted_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that loading packed datasets works correctly."""
    check_loaded_datasets(load_datasets(compacted_test_datasets))


def test_load_valid_packed_generated_datasets(generated_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that loading packed generated datasets works correctly."""
    check_loaded_datasets(load_datasets(generated_test_datasets), SampleStage.GENERATED)


def test_load_valid_packed_evaluated_datasets(evaluated_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that loading packed evaluated datasets works correctly."""
    check_loaded_datasets(load_datasets(evaluated_test_datasets), SampleStage.EVALUATED)


def test_load_valid_unpacked_datasets_with_gitignore(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check loading unpacked datasets with `.gitignore` files works correctly."""
    # Create a `.gitignore` file which ignores any `obj/` directories
    gitignore_path = expanded_test_datasets / ".gitignore"
    gitignore_path.write_text("obj/\n")
    # Add some files in `obj/` directories to the dataset
    ada_sample_0_dir = expanded_test_datasets / "ada_test" / "test_sample_0"
    spark_sample_1_dir = expanded_test_datasets / "spark_test" / "test_sample_1"
    for source_dir in (
        ada_sample_0_dir / "base",
        spark_sample_1_dir / "solution" / "source_dir_1",
    ):
        (source_dir / "obj").mkdir(exist_ok=True)
        (source_dir / "obj" / "some_file").write_text("This is a test file.\n")

    # Load and check the datasets
    datasets = load_datasets(expanded_test_datasets)

    # Check that the `obj/some_file` files are present in the loaded datasets
    # because `expanded_test_dataset` is not in a Git worktree.
    ada_dataset = next(d for d in datasets if d.sample_type is AdaSample)
    assert ada_dataset.samples[0].name == "test_sample_0"
    ada_ignored_rel_path = Path("obj/some_file")
    assert ada_ignored_rel_path in ada_dataset.samples[0].sources.files
    assert ada_dataset.samples[0].sources.files[ada_ignored_rel_path] == (
        "This is a test file.\n"
    )
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    spark_sample_1 = next(s for s in spark_dataset.samples if s.name == "test_sample_1")
    spark_ignored_rel_path = Path("source_dir_1/obj/some_file")
    assert spark_ignored_rel_path in spark_sample_1.canonical_solution.files
    assert spark_sample_1.canonical_solution.files[spark_ignored_rel_path] == (
        "This is a test file.\n"
    )
    # The loaded datasets should otherwise be as expected
    ada_dataset.samples[0].sources.files.pop(ada_ignored_rel_path)
    spark_sample_1.canonical_solution.files.pop(spark_ignored_rel_path)
    check_loaded_datasets(datasets)

    # Initialise a Git repository in the dataset directory and reload
    setup_git_repo(expanded_test_datasets)
    datasets = load_datasets(expanded_test_datasets)

    # Check that this time the `obj/some_file` files are not present in the
    # loaded datasets
    ada_dataset = next(d for d in datasets if d.sample_type is AdaSample)
    assert ada_dataset.samples[0].name == "test_sample_0"
    assert ada_ignored_rel_path not in ada_dataset.samples[0].sources.files
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    spark_sample_1 = next(s for s in spark_dataset.samples if s.name == "test_sample_1")
    assert spark_ignored_rel_path not in spark_sample_1.canonical_solution.files
    check_loaded_datasets(datasets)


def test_load_empty_packed_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
):
    """Check that loading an empty jsonl file yields an appropriate empty dataset."""
    empty_dataset_path = tmp_path / "ada_empty.jsonl"
    empty_dataset_path.touch()
    datasets = load_datasets(empty_dataset_path)
    assert len(datasets) == 1
    empty_dataset = datasets[0]
    assert empty_dataset.name == "empty"
    assert empty_dataset.sample_type is EvaluatedAdaSample
    assert empty_dataset.samples == []
    assert_log(caplog, WARN, f"Dataset at '{empty_dataset_path}' is empty.")
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_load_no_valid_samples(
    compacted_test_datasets,  # noqa: F811  # pytest fixture
    expanded_test_datasets,  # noqa: F811  # pytest fixture
    caplog: pytest.LogCaptureFixture,
):
    """Check that loading a dataset with no valid samples produces a warning."""
    # Remove the `.jsonl` suffix from all packed datasets and check that loading
    # them issues a warning
    for packed_dataset in compacted_test_datasets.iterdir():
        packed_dataset.rename(packed_dataset.with_suffix(""))
    datasets = load_datasets(compacted_test_datasets)
    assert len(datasets) == 0
    assert_log(
        caplog, WARN, f"No datasets could be found at: {compacted_test_datasets}"
    )
    # Remove the `other.json` file from all unpacked datasets and check that
    # loading them issues a warning
    for other_json in expanded_test_datasets.glob("**/other.json"):
        other_json.unlink()
    datasets = load_datasets(expanded_test_datasets)
    assert len(datasets) == 0
    assert_log(caplog, WARN, f"No datasets could be found at: {expanded_test_datasets}")


def test_load_mixed_datasets(
    tmp_path: Path,
    compacted_test_datasets,  # noqa: F811  # pytest fixture
    expanded_test_datasets,  # noqa: F811  # pytest fixture
):
    """Check that loading directory with mixed packed/unpacked datasets gives error."""
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    for fixture_dir in (compacted_test_datasets, expanded_test_datasets):
        for dataset in fixture_dir.iterdir():
            shutil.move(dataset, mixed_dir / (dataset.relative_to(fixture_dir)))
    error_msg = f"'{mixed_dir}' contains a mixture of packed and unpacked datasets."
    with pytest.raises(MixedDatasetFormatsError, match=re.escape(error_msg)):
        load_datasets(mixed_dir)


def test_load_invalid_samples(
    compacted_test_datasets,  # noqa: F811  # pytest fixture
    expanded_test_datasets,  # noqa: F811  # pytest fixture
):
    """Test that exceptions while loading samples specify which sample raised them."""
    # Make one of the samples in `spark_test.jsonl` invalid
    spark_test_path = compacted_test_datasets / "spark_test.jsonl"
    original_spark_test = spark_test_path.read_text()
    spark_test_path.write_text(
        original_spark_test.replace('"name":"test_sample_1",', "")
    )
    error_msg = r"^1 validation error for SparkSample\nname\n  Field required .*\n.*\n"
    error_msg += re.escape(
        f"This error occurred while parsing line 2 of '{spark_test_path}'"
    )
    with pytest.raises(pydantic.ValidationError, match=error_msg):
        load_datasets(compacted_test_datasets)

    # Make an `other.json` file invalid JSON and check the resulting error
    sample_dir = expanded_test_datasets / "explain_test" / "test_sample_0"
    other_json_path = sample_dir / "other.json"
    original_other_json = other_json_path.read_text()
    other_json_path.write_text("This is not valid JSON")
    location_note = (
        f"\nThis exception occurred while loading the sample at: {sample_dir}"
    )
    error_msg = "Expecting value: line 1 column 1 (char 0)" + location_note
    with pytest.raises(json.decoder.JSONDecodeError, match=re.escape(error_msg)):
        load_datasets(expanded_test_datasets)
    # Restore the `other.json` file, but with an invalid `Location`
    other_json_invalid_location = json.loads(original_other_json)
    other_json_invalid_location["location"].pop("path")
    other_json_path.write_text(json.dumps(other_json_invalid_location))
    error_msg = r"^1 validation error for Location\npath\n  Field required .*\n.*"
    error_msg += re.escape(location_note)
    with pytest.raises(pydantic.ValidationError, match=error_msg):
        load_datasets(expanded_test_datasets)


def test_load_invalid_sample_name(
    compacted_test_datasets,  # noqa: F811  # pytest fixture
    expanded_test_datasets,  # noqa: F811  # pytest fixture
):
    """Check that loading a dataset with an invalid sample name raises an error."""
    # Check packed
    spark_test_path = compacted_test_datasets / "spark_test.jsonl"
    original_spark_test = spark_test_path.read_text()
    spark_test_path.write_text(
        original_spark_test.replace('"name":"test_sample_1"', '"name":"test_sample_#1"')
    )
    error_msg = re.escape(
        "1 validation error for SparkSample\nname\n"
        "  Value error, Invalid sample name: 'test_sample_#1'. Please only use "
        "alphanumeric characters, hyphens, and underscores. "
    )
    error_msg += r".*\n.*\n"
    error_msg += re.escape(
        f"This error occurred while parsing line 2 of '{spark_test_path}'"
    )
    with pytest.raises(pydantic.ValidationError, match=error_msg):
        load_datasets(compacted_test_datasets)

    # Check unpacked
    ada_sample_path = expanded_test_datasets / "ada_test" / "test_sample_0"
    ada_sample_path = ada_sample_path.rename(ada_sample_path.parent / "test_sample_#0")
    error_msg = re.escape(
        "1 validation error for Sample\nname\n"
        "  Value error, Invalid sample name: 'test_sample_#0'. Please only use "
        "alphanumeric characters, hyphens, and underscores. "
    )
    error_msg += r".*\n.*\n"
    error_msg = (
        f"This exception occurred while loading the sample at: {ada_sample_path}"
    )
    with pytest.raises(pydantic.ValidationError, match=error_msg):
        load_datasets(expanded_test_datasets)


def test_load_non_sample_warning(
    expanded_test_datasets: Path,  # noqa: F811  # pytest fixture
    caplog: pytest.LogCaptureFixture,
):
    """Check that a non-sample file/directory is ignored with a warning."""
    # Remove the `other.json` file from one of the spark samples and check that
    # loading the dataset issues a warning
    spark_sample_path = expanded_test_datasets / "spark_test" / "test_sample_0"
    (spark_sample_path / "other.json").unlink()
    datasets = load_datasets(expanded_test_datasets)
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    assert len(spark_dataset.samples) == 2
    assert_log(caplog, WARN, f"Skipping non-sample directory: {spark_sample_path}")


def test_load_invalid_dataset_name(compacted_test_datasets, expanded_test_datasets):  # noqa: F811  # pytest fixtures
    """Check that loading a dataset with an invalid name format raises an error."""
    shutil.move(
        compacted_test_datasets / "ada_test.jsonl",
        compacted_test_datasets / "ada test.jsonl",
    )
    error_msg = "Expected packed dataset filename to contain an underscore:"
    with pytest.raises(InvalidDatasetNameError, match=error_msg):
        load_datasets(compacted_test_datasets)
    shutil.move(
        expanded_test_datasets / "ada_test",
        expanded_test_datasets / "ada test",
    )
    error_msg = "Expected unpacked dataset dir name to contain an underscore:"
    with pytest.raises(InvalidDatasetNameError, match=error_msg):
        load_datasets(expanded_test_datasets)


def test_load_invalid_dataset_kind(compacted_test_datasets, expanded_test_datasets):  # noqa: F811  # pytest fixtures
    """Check that loading a dataset with an invalid kind raises an error."""
    shutil.move(
        compacted_test_datasets / "ada_test.jsonl",
        compacted_test_datasets / "unknown_test.jsonl",
    )
    error_msg = "Unknown dataset kind: unknown"
    with pytest.raises(UnknownDatasetKindError, match=re.escape(error_msg)):
        load_datasets(compacted_test_datasets)
    shutil.move(
        expanded_test_datasets / "ada_test",
        expanded_test_datasets / "unknown_test",
    )
    with pytest.raises(UnknownDatasetKindError, match=re.escape(error_msg)):
        load_datasets(expanded_test_datasets)


def test_load_duplicate_sample_names(compacted_test_datasets):  # noqa: F811  # pytest fixture
    """Check that loading a dataset with duplicate sample names raises an error."""
    spark_dataset_file = compacted_test_datasets / "spark_test.jsonl"
    spark_dataset_packed_content = spark_dataset_file.read_text()
    spark_dataset_packed_content = spark_dataset_packed_content.replace(
        '"name":"test_sample_0"', '"name":"test_sample_1"'
    )
    spark_dataset_file.write_text(spark_dataset_packed_content)
    error_msg = f"Duplicate sample name 'test_sample_1' found in '{spark_dataset_file}'"
    with pytest.raises(DuplicateSampleNameError, match=re.escape(error_msg)):
        load_datasets(compacted_test_datasets)


def test_load_mixed_sample_types(generated_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that loading a dataset with mixed sample types raises an error."""
    spark_dataset_file = generated_test_datasets / "spark_test.jsonl"
    spark_dataset_content = spark_dataset_file.read_text()
    # Change one of the generated samples to an evaluated one
    spark_dataset_content = spark_dataset_content.replace(
        '"name":"test_sample_2"',
        '"name":"test_sample_2","evaluation_results":[]',
    )
    spark_dataset_file.write_text(spark_dataset_content)
    error_msg = (
        f"Dataset at '{spark_dataset_file}' contains mixed sample types:\n"
        f"'test_sample_0' is GeneratedSparkSample "
        f"but 'test_sample_2' is EvaluatedSparkSample"
    )
    with pytest.raises(MixedSampleTypesError, match=re.escape(error_msg)):
        load_datasets(generated_test_datasets)


def test_load_unpacked_dataset_invalid(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that `load_unpacked_dataset()` defensively raises on an invalid dataset."""
    # Remove the `other.json` file from the ada dataset
    ada_dataset_path = expanded_test_datasets / "ada_test"
    (ada_dataset_path / "test_sample_0" / "other.json").unlink()
    # Attempting to load this dataset directly with `load_unpacked_dataset`
    # should raise an `InvalidDatasetError`
    error_msg = f"'{ada_dataset_path}' is not a valid unpacked dataset"
    with pytest.raises(InvalidDatasetError, match=re.escape(error_msg)):
        load_unpacked_dataset(ada_dataset_path)


def test_load_packed_dataset_invalid(compacted_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that `load_packed_dataset()` defensively raises on an invalid dataset."""
    # Remove the `.jsonl` suffix from the ada dataset
    ada_dataset_path = compacted_test_datasets / "ada_test.jsonl"
    ada_dataset_path = ada_dataset_path.rename(compacted_test_datasets / "ada_test")
    # Attempting to load this dataset directly with `load_packed_dataset`
    # should raise an `InvalidDatasetError`
    error_msg = f"'{ada_dataset_path}' is not a valid packed dataset"
    with pytest.raises(InvalidDatasetError, match=re.escape(error_msg)):
        load_packed_dataset(ada_dataset_path)
