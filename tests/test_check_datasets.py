import re
import shutil
from logging import ERROR, INFO
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from helpers import assert_log

from ada_eval.check_datasets import (
    FailedCanonicalEvaluationError,
    InaccurateCanonicalEvaluationError,
    InvalidBaselineEvaluationError,
    PassedBaselineEvaluationError,
    check_base_datasets,
)
from ada_eval.datasets import (
    Dataset,
    Eval,
    EvaluationStatsBuild,
    Sample,
    dataset_has_sample_type,
    load_datasets,
    save_datasets,
)
from ada_eval.datasets.types.datasets import DatasetsMismatchError
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.samples import (
    EvaluatedSparkSample,
    ExplainSample,
    GeneratedSparkSample,
    GenerationStats,
    SparkSample,
)
from ada_eval.evals.evaluate import create_eval
from ada_eval.evals.generic_eval import GenericEval


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprclean"), reason="gprclean not available")
@pytest.mark.skipif(not shutil.which("gnatformat"), reason="gnatformat not available")
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_check_base_datasets(
    tmp_path: Path,
    check_test_datasets: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    caplog.set_level("INFO")

    # Load datasets consisting of one `SparkSample` which passes all checks, one
    # bad `SparkSample`, and one `ExplainSample` (which has some canonical
    # results it shouldn't, but is otherwise correct).
    datasets = load_datasets(check_test_datasets)
    explain_dataset = next(
        d for d in datasets if dataset_has_sample_type(d, ExplainSample)
    )
    explain_sample = explain_dataset.samples[0]
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    good_sample = next(s for s in spark_dataset.samples if s.name == "good")
    bad_sample = next(s for s in spark_dataset.samples if s.name == "bad")

    # Save this dataset in both expanded and compacted forms
    expanded_dir = tmp_path / "expanded"
    compacted_dir = tmp_path / "compacted"

    def save_both(
        expanded: list[Dataset[Sample]], compacted: list[Dataset[Sample]] | None = None
    ) -> None:
        if compacted is None:
            compacted = expanded
        save_datasets(expanded, expanded_dir, unpacked=True)
        save_datasets(compacted, compacted_dir, unpacked=False)

    save_both(datasets)

    # Check that a missing dataset is detected
    def run_check() -> None:
        check_base_datasets([expanded_dir, compacted_dir], jobs=8)

    shutil.copytree(expanded_dir / "spark_check", expanded_dir / "spark_other")
    error_msg = f"dataset 'spark_other' is only present in '{expanded_dir}'."
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

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
    save_both([spark_dataset, explain_dataset], [explain_dataset, generated_dataset])
    error_msg = (
        f"dataset 'spark_check' has type 'SparkSample' in '{expanded_dir}' but "
        f"type 'GeneratedSparkSample' in '{compacted_dir}'."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that missing samples are detected
    one_sample_dataset = Dataset(
        name="check",
        sample_type=SparkSample,
        samples=[good_sample],
    )
    save_both([explain_dataset, one_sample_dataset], [spark_dataset, explain_dataset])
    error_msg = (
        f"sample 'bad' in dataset 'spark_check' is only present in '{compacted_dir}'."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that differing samples are detected
    modified_sample_0 = good_sample.model_copy(deep=True)
    modified_sample_0.canonical_solution.files[Path("src/foo.adb")] = b"expanded nested"
    modified_sample_0.sources.files[Path("new_file")] = b"new"
    modified_sample_0.prompt = "Modified prompt"
    modified_dataset_0 = Dataset(
        name="check", sample_type=SparkSample, samples=[bad_sample, modified_sample_0]
    )
    modified_sample_1 = good_sample.model_copy(deep=True)
    modified_sample_1.canonical_solution.files[Path("src/foo.adb")] = (
        b"compacted nested"
    )
    modified_dataset_1 = Dataset(
        name="check", sample_type=SparkSample, samples=[modified_sample_1, bad_sample]
    )
    save_both([modified_dataset_0], [modified_dataset_1])
    error_msg = (
        "sample 'good' in dataset 'spark_check' differs between "
        f"'{expanded_dir}' and '{compacted_dir}':\n\n"
        "{'prompt': 'Modified prompt', 'sources': {PosixPath('new_file'): b'new'},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): b'expanded nested'}}"
        "\n\n{'prompt': '', 'sources': {},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): b'compacted nested'}}"
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that non-passing canonical evaluation results are detected
    save_both(datasets)
    error_msg = (
        "sample 'bad' in dataset 'spark_check' has non-passing canonical "
        "evaluation results: ['prove', 'build', 'test']"
    )
    with pytest.raises(FailedCanonicalEvaluationError, match=re.escape(error_msg)):
        run_check()

    # Fix one of the canonical results and check that the message changes
    # appropriately
    bad_sample_build_stats = bad_sample.canonical_evaluation_results[1]
    assert isinstance(bad_sample_build_stats, EvaluationStatsBuild)
    bad_sample_build_stats.pre_format_warnings = False
    save_both(datasets)
    error_msg = (
        "sample 'bad' in dataset 'spark_check' has non-passing canonical "
        "evaluation results: ['prove', 'test']"
    )
    with pytest.raises(FailedCanonicalEvaluationError, match=re.escape(error_msg)):
        run_check()

    # Nothing should have been output or logged up to this point
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert caplog.text == ""

    # Fix the remaining canonical results and check that the discrepancy is
    # detected by re-evaluating
    bad_sample.canonical_evaluation_results = good_sample.canonical_evaluation_results
    save_both(datasets)
    error_msg = re.escape(
        "mismatch found on re-evaluating sample 'bad' in dataset 'spark_check':\n\n"
        "[{'pre_format_warnings': False}, {'result': 'proved', 'proved_checks': "
        "{'VC_POSTCONDITION': 1}, 'unproved_checks': {}}, {'passed_tests': True}]\n\n"
        "[{'pre_format_warnings': True}, {'result': 'unproved', 'proved_checks': {}, "
        "'unproved_checks': {'VC_POSTCONDITION': 1}}, {'passed_tests': False}]"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()

    # There should have been an info log and a loading bar for each evaluation
    reeval_msg = "Re-evaluating to check canonical evaluation results are accurate ..."
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()
    output = capsys.readouterr()
    for e in ("build", "prove", "test"):
        assert f"Evaluating with {e}" in output.err
    assert output.out == ""

    # Check that a missing canonical result yields a more readable error message
    bad_sample.canonical_evaluation_results = [
        good_sample.canonical_evaluation_results[1]
    ]
    save_both(datasets)
    error_msg = re.escape(
        "sample 'bad' in dataset 'spark_check' does not have the expected set of "
        "canonical evaluation results:\n['build'] != ['build', 'prove', 'test']"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()

    # Fix the actual issues with the bad sample's canonical solution and check
    # that the presence of a test result for the explain sample is detected.
    bad_sample.canonical_evaluation_results = good_sample.canonical_evaluation_results
    bad_sample.canonical_solution = good_sample.canonical_solution
    save_both(datasets)
    error_msg = re.escape(
        "sample 'explain' in dataset 'explain_check' does not have the "
        "expected set of canonical evaluation results:\n['test'] != []"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()

    # Clear the `ExplainSample`'s canonical results and check that the passing
    # baseline evaluation of the 'bad' sample is detected
    explain_sample.canonical_evaluation_results = []
    save_both(datasets)
    error_msg = re.escape(
        "all evaluations passed on the unmodified sources of sample 'bad' in "
        "dataset 'spark_check'."
    )
    with pytest.raises(PassedBaselineEvaluationError, match=error_msg):
        run_check()
    baseline_msg = "Checking that null generations fail at least one eval ..."
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert len(caplog.records) == 2
    caplog.clear()

    # Modify the bad sample's `sources` to fail the evaluations and check that
    # `check_base_datasets()` raises no exceptions
    bad_sample.sources = good_sample.sources
    save_both(datasets)
    run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert_log(caplog, INFO, "Base datasets are correct.")
    assert len(caplog.records) == 3
    caplog.clear()

    # Mock `create_eval()` so that an exception is raised during evaluation of
    # the base sources (but not the canonical solutions), and check that this is
    # detected
    class MockEval(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
        """Mock build eval which raises if `src/foo.adb` contains 'return 1'."""

        eval: ClassVar = Eval.BUILD
        supported_types: ClassVar = {GeneratedSparkSample: EvaluatedSparkSample}

        def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsBuild:
            if b"return 1" in sample.generated_solution.files[Path("src/foo.adb")]:
                raise RuntimeError("Mock evaluation error")
            return EvaluationStatsBuild(
                compiled=True, pre_format_warnings=False, post_format_warnings=False
            )

    def mock_create_eval(evaluation: Eval):
        if evaluation == Eval.BUILD:
            return MockEval()
        return create_eval(evaluation)

    error_msg = (
        "error during baseline evaluation of sample 'bad' in dataset 'spark_check': "
        "EvaluationStatsFailed(eval=<Eval.BUILD: 'build'>, "
        "exception=\"RuntimeError('Mock evaluation error')\")"
    )
    with (
        patch("ada_eval.evals.evaluate.create_eval", mock_create_eval),
        pytest.raises(InvalidBaselineEvaluationError, match=re.escape(error_msg)),
    ):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert_log(caplog, ERROR, "Error during evaluation of sample bad")
    assert_log(caplog, ERROR, "Error during evaluation of sample good")
    assert len(caplog.records) == 4
    caplog.clear()

    # Test checking an empty list of dataset directories
    check_base_datasets([], jobs=8)
    assert_log(caplog, INFO, "No dataset directories specified; nothing to check.")
    assert len(caplog.records) == 1
