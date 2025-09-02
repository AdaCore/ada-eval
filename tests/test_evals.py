import os
import re
import shutil
import subprocess
from logging import ERROR, WARN
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

import pytest
from helpers import (
    assert_git_status,
    assert_log,
    eval_test_datasets,  # noqa: F401  # pytest fixture
    evaluated_test_datasets,  # noqa: F401  # pytest fixture
    expanded_test_datasets,  # noqa: F401  # pytest fixture
    generated_test_datasets,  # noqa: F401  # pytest fixture
    setup_git_repo,
)

from ada_eval.datasets import Dataset, dataset_has_sample_type
from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types.evaluation_stats import (
    Eval,
    EvaluationStatsBase,
    EvaluationStatsBuild,
    EvaluationStatsFailed,
    EvaluationStatsProve,
    EvaluationStatsTimedOut,
)
from ada_eval.datasets.types.samples import (
    EVALUATED_SAMPLE_TYPES,
    GENERATED_SAMPLE_TYPES,
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
)
from ada_eval.evals.generic_eval import GenericEval, WrongEvalOutputTypeError
from ada_eval.evaluate import (
    evaluate_datasets,
    evaluate_datasets_canonical,
    evaluate_directory,
)
from ada_eval.utils import ExecutableNotFoundError

GENERATED_TYPE_TO_EVALUATED = {
    GENERATED_SAMPLE_TYPES[k]: EVALUATED_SAMPLE_TYPES[k] for k in SampleKind
}


# Mock eval stats matching those in `evaluated_test_datasets`
MOCK_BUILD_EVAL_STATS = EvaluationStatsBuild(
    compiled=False, pre_format_warnings=True, post_format_warnings=True
)
MOCK_PROVE_EVAL_STATS = EvaluationStatsProve(
    successfully_proven=False, subprogram_found=True
)


class MockBuildEval(GenericEval[GeneratedSample, EvaluatedSample]):
    """Mock build eval suitable for reproducing `evaluated_test_datasets`."""

    eval: ClassVar = Eval.BUILD
    supported_types: ClassVar = GENERATED_TYPE_TO_EVALUATED

    def evaluate(self, _: GeneratedSample) -> EvaluationStatsBuild:
        return MOCK_BUILD_EVAL_STATS


class MockProveEval(GenericEval[GeneratedSample, EvaluatedSample]):
    """Mock prove eval suitable for reproducing `evaluated_test_datasets`."""

    eval: ClassVar = Eval.PROVE
    supported_types: ClassVar = GENERATED_TYPE_TO_EVALUATED

    def evaluate(self, _: GeneratedSample) -> EvaluationStatsProve:
        return MOCK_PROVE_EVAL_STATS


def mock_create_eval(eval_: Eval) -> MockBuildEval | MockProveEval:
    """Mock `create_eval()` suitable for reproducing `evaluated_test_datasets`."""
    match eval_:
        case Eval.BUILD:
            return MockBuildEval()
        case Eval.PROVE:
            return MockProveEval()
        case _:
            raise ValueError(f"Unknown mock eval {eval_}")


def check_progress_bar(output: Any, total: int, eval_name: str):
    assert f"Evaluating with {eval_name}: 100%" in output.err
    assert f" {total}/{total} " in output.err
    assert output.out == ""


def test_generic_eval(
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Create mock evaluation stats and two evals. Both evals just return the
    # mock stats, but `MockEval1` is incompatible with `ExplainSample`s, raises
    # a subprocess timeout on "test_sample_1", and raises a subprocess error on
    # "test_sample_2".
    class MockEvaluationStats(EvaluationStatsBase):
        pass

    class MockEval0(GenericEval[GeneratedSample, EvaluatedSample]):
        eval: ClassVar = Eval.BUILD
        supported_types: ClassVar = {
            GeneratedAdaSample: EvaluatedAdaSample,
            GeneratedExplainSample: EvaluatedExplainSample,
            GeneratedSparkSample: EvaluatedSparkSample,
        }

        def evaluate(self, _: GeneratedSample) -> MockEvaluationStats:  # type: ignore[override]  # `MockEvaluationStats` is not a real `EvaluationStats`
            return MockEvaluationStats(eval=Eval.BUILD)

    class MockEval1(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
        eval: ClassVar = Eval.PROVE
        supported_types: ClassVar = {
            GeneratedAdaSample: EvaluatedAdaSample,
            GeneratedSparkSample: EvaluatedSparkSample,
        }

        def evaluate(self, sample: GeneratedAdaSample) -> MockEvaluationStats:  # type: ignore[override]  # `MockEvaluationStats` is not a real `EvaluationStats`
            if sample.name == "test_sample_1":
                raise subprocess.TimeoutExpired(
                    cmd=["cmd", "timeout-arg"],
                    timeout=1.2,
                    output=b"This is the stdout",
                    stderr=b"This is the stderr",
                )
            if sample.name == "test_sample_2":
                raise subprocess.CalledProcessError(
                    returncode=42,
                    cmd=["cmd", "fail-arg"],
                    output="This is a\nmulti-line\nstdout",
                    stderr="This is the stderr",
                )
            return MockEvaluationStats(eval=Eval.PROVE)

    # Also define an eval which is compatible with nothing, to check that it is
    # effectively ignored.
    class MockEval2(GenericEval[GeneratedSample, EvaluatedSample]):
        eval: ClassVar = Eval.PROVE
        supported_types: ClassVar = {}

        def evaluate(self, _) -> MockEvaluationStats:  # type: ignore[override]  # `MockEvaluationStats` is not a real `EvaluationStats`
            return MockEvaluationStats(eval="invalid")

    # Define a mock `create_eval` function which takes the mock eval ID instead
    # of a real `Eval`.
    def mock_create_eval(eval_id: int) -> MockEval0 | MockEval1 | MockEval2:
        match eval_id:
            case 0:
                return MockEval0()
            case 1:
                return MockEval1()
            case 2:
                return MockEval2()
            case _:
                raise ValueError(f"Unknown mock eval ID {eval_id}")

    # Run the eval on the generated test datasets.
    generated_datasets = cast(
        list[Dataset[GeneratedSample]], load_datasets(generated_test_datasets)
    )
    original_samples = {
        (d.dirname(), s.name): s for d in generated_datasets for s in d.samples
    }
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluated_datasets = evaluate_datasets(
            evals=[0, 1, 2],  # type: ignore[list-item]  # Using mock IDs instead of real enum
            datasets=generated_datasets,
            jobs=8,
        )

    def mock_1_expected_stats(sample: Sample) -> list[EvaluationStatsBase]:
        if isinstance(sample, ExplainSample):
            # Incompatible with `ExplainSample`s
            return []
        if sample.name == "test_sample_1":
            # Raises a timeout on samples called "test_sample_1"
            return [
                EvaluationStatsTimedOut(
                    eval=Eval.PROVE, cmd_timed_out=["cmd", "timeout-arg"], timeout=1.2
                )
            ]
        if sample.name == "test_sample_2":
            # Raises an exception on samples called "test_sample_2"
            return [
                EvaluationStatsFailed(eval=Eval.PROVE, exception="CalledProcessError()")
            ]
        # Otherwise evaluates successfully
        return [MockEvaluationStats(eval=Eval.PROVE)]

    def check_evaluated_datasets(datasets: list[Dataset[EvaluatedSample]]):
        for dataset in datasets:
            assert dataset_has_sample_type(dataset, EvaluatedSample)
            for sample in dataset.samples:
                generated_sample = original_samples[(dataset.dirname(), sample.name)]
                assert isinstance(sample, EvaluatedSample)
                assert sample.evaluation_results == [
                    MockEvaluationStats(eval=Eval.BUILD),
                    *mock_1_expected_stats(generated_sample),
                ]
                assert generated_sample.model_dump() == sample.model_dump(
                    exclude={"evaluation_results"}
                )

    # Check that the evaluated samples were created correctly, with suitable
    # progress bars
    check_evaluated_datasets(evaluated_datasets)
    output = capsys.readouterr()
    check_progress_bar(output, 5, "build")  # 5 total samples (1+1+3)
    check_progress_bar(output, 4, "prove")  # 4 total samples (1+3)
    assert len(evaluated_datasets) == 3
    assert sum(len(d.samples) for d in evaluated_datasets) == 5

    # The failure should have been logged with a full stack trace, with
    # additional notes containing the stdout and stderr.
    fail_log = assert_log(
        caplog, ERROR, "Error during evaluation of sample test_sample_2"
    )
    assert fail_log.exc_text.endswith(
        "subprocess.CalledProcessError: "
        "Command '['cmd', 'fail-arg']' returned non-zero exit status 42.\n"
        "stdout: 'This is a\\nmulti-line\\nstdout'\n"
        "stderr: 'This is the stderr'"
    )
    # The timeout should have been logged with a warning but no stack trace
    warning = (
        "Evaluation of sample test_sample_1 failed due to subprocess timeout "
        "(1.2 seconds)"
    )
    timeout_log = assert_log(caplog, WARN, warning)
    assert timeout_log.exc_text is None
    # The eval which is compatible with nothing should have logged a warning
    assert_log(caplog, WARN, "No datasets compatible with prove found.")

    # Run the mock evals as a canonical evaluation on the evaluated datasets
    # (canonical evaluations would usually be run on base datasets, but this
    # should work too, and serves to check that the results of evaluating the
    # generations do not pollute the canonical results).
    #
    # Note the order of evals is reversed.
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluated_datasets = evaluate_datasets_canonical(
            evals=[2, 1, 0],  # type: ignore[list-item]  # Using mock IDs instead of real enum
            datasets=list(evaluated_datasets),
            jobs=8,
        )
    # Check that the `MockEvaluationStat`s were merged into the existing
    # `canonical_evaluation_results`, then restore the originals so that
    # `check_evaluated_datasets()` can verify the rest.
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            original_sample = original_samples[(dataset.dirname(), sample.name)]
            assert isinstance(sample, EvaluatedSample)
            # The existing eval ordering should be preserved (which makes the
            # diff cleaner)
            assert all(
                es1.eval == es2.eval
                for es1, es2 in zip(
                    sample.canonical_evaluation_results,
                    original_sample.canonical_evaluation_results,
                    strict=False,  # Not all original samples have both eval results
                )
            )
            results: dict[Eval, EvaluationStatsBase] = {
                es.eval: es for es in sample.canonical_evaluation_results
            }
            if isinstance(sample, EvaluatedExplainSample):
                # `MockEval1` is incompatible with `ExplainSample`s, so only
                # the results from `MockEval0` should be present.
                assert results[Eval.BUILD] == MockEvaluationStats(eval=Eval.BUILD)
            else:
                # Both evals are compatible, so the results from both should
                # be present.
                assert results[Eval.BUILD] == MockEvaluationStats(eval=Eval.BUILD)
                assert results[Eval.PROVE] == mock_1_expected_stats(sample)[0]
            # Restore the original canonical results for `check_evaluated_datasets()`
            sample.canonical_evaluation_results = (
                original_sample.canonical_evaluation_results
            )
    # Check that everything else is still correct
    check_evaluated_datasets(evaluated_datasets)
    assert len(evaluated_datasets) == 3
    assert sum(len(d.samples) for d in evaluated_datasets) == 5


def test_generic_eval_wrong_output_type(
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Test that an exception is raised if `supported_types` is misconfigured."""

    # Create a mock eval with a misconfigured `supported_types` (mapping
    # `AdaSample` to `ExplainSample`).
    class MockEval(GenericEval[GeneratedSample, EvaluatedSample]):
        eval: ClassVar = Eval.BUILD
        supported_types: ClassVar = {
            GeneratedAdaSample: EvaluatedExplainSample,
            GeneratedSparkSample: EvaluatedSparkSample,
        }

        def evaluate(self, _: GeneratedSample) -> EvaluationStatsFailed:
            # Use EvaluationStatsFailed as a dummy return value
            return EvaluationStatsFailed(eval=Eval.BUILD, exception="")

    # Check that running this eval on datasets including an `AdaSample` raises
    # a `WrongEvalOutputTypeError`.
    error_msg = (
        "Eval 'build' accepted a GeneratedSample of type GeneratedAdaSample, "
        "but the corresponding evaluated sample type (EvaluatedAdaSample) is "
        "not compatible with the eval's output types "
        "(EvaluatedExplainSample, EvaluatedSparkSample)."
    )
    with (
        patch("ada_eval.evaluate.create_eval", return_value=MockEval()),
        pytest.raises(WrongEvalOutputTypeError, match=re.escape(error_msg)),
    ):
        evaluate_datasets(
            evals=[None],  # type: ignore[list-item]  # Dummy value; just need length 1
            datasets=cast(
                list[Dataset[GeneratedSample]], load_datasets(generated_test_datasets)
            ),
            jobs=8,
        )

    # Nothing else should have been logged
    assert caplog.text == ""
    output = capsys.readouterr()
    assert output.out == ""
    assert "Evaluating with build:" in output.err


def test_evaluate_datasets_no_evals(
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    caplog: pytest.LogCaptureFixture,
):
    """Test that `evaluate_datasets()` warns when no `Eval`s are provided."""
    with patch("ada_eval.evaluate.create_eval") as mock_create_eval:
        datasets = cast(
            list[Dataset[GeneratedSample]], load_datasets(generated_test_datasets)
        )
        returned_datasets = evaluate_datasets(evals=[], datasets=datasets, jobs=8)
    assert returned_datasets == []
    assert not mock_create_eval.called
    assert_log(caplog, WARN, "No evals provided; skipping evaluation.")


def test_evaluate_directory(
    tmp_path: Path,
    evaluated_test_datasets: Path,  # noqa: F811  # pytest fixture
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Init a Git repo to track changes.
    setup_git_repo(tmp_path, initial_commit=True)
    assert_git_status(tmp_path, expect_dirty=False)

    # Delete the contents of the evaluated datasets directory.
    assert (evaluated_test_datasets / "spark_test.jsonl").is_file()
    shutil.rmtree(evaluated_test_datasets)
    assert not evaluated_test_datasets.exists()
    assert_git_status(tmp_path, expect_dirty=True)

    # Run the evals on the generated test datasets and check this regenerates
    # the evaluated datasets.
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.PROVE, Eval.BUILD],
            path=generated_test_datasets,
            output_dir=evaluated_test_datasets,
            jobs=8,
        )
    assert (evaluated_test_datasets / "spark_test.jsonl").is_file()
    assert_git_status(tmp_path, expect_dirty=False)

    # Only the progress bars should be output.
    assert caplog.text == ""
    output = capsys.readouterr()
    check_progress_bar(output, 5, "build")
    check_progress_bar(output, 5, "prove")

    # Load a copy of the original generated datasets for later comparison.
    original_generated_datasets = load_datasets(generated_test_datasets)
    original_samples = {
        (d.dirname(), s.name): s for d in original_generated_datasets for s in d.samples
    }

    # Run the evals as a canonical evaluation on the generated datasets
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD, Eval.PROVE],
            path=generated_test_datasets,
            output_dir=generated_test_datasets,
            jobs=8,
            canonical_evaluation=True,
        )

    # The output should be saved in packed format
    assert (generated_test_datasets / "spark_test.jsonl").is_file()
    assert not (generated_test_datasets / "spark_test").exists()

    # Check that the `canonical_evaluation_results` have been updated correctly:
    # the mock results should replace the existing ones, as they have the same
    # `eval` field values.
    canonically_evaluated_datasets = load_datasets(generated_test_datasets)
    for dataset in canonically_evaluated_datasets:
        assert dataset_has_sample_type(dataset, GeneratedSample)
        for sample in dataset.samples:
            original_sample = original_samples[(dataset.dirname(), sample.name)]
            assert isinstance(sample, GeneratedSample)
            # The existing ordering will be preserved (which makes the diff
            # cleaner)
            assert all(
                es1.eval == es2.eval
                for es1, es2 in zip(
                    sample.canonical_evaluation_results,
                    original_sample.canonical_evaluation_results,
                    strict=False,  # Not all original samples have both eval results
                )
            )
            assert sorted(
                sample.canonical_evaluation_results, key=lambda x: x.eval
            ) == [MOCK_BUILD_EVAL_STATS, MOCK_PROVE_EVAL_STATS]
            # The remaining fields should be unchanged
            assert sample.model_dump(
                exclude={"canonical_evaluation_results"}
            ) == original_sample.model_dump(exclude={"canonical_evaluation_results"})

    # The output should be the same as before
    assert caplog.text == ""
    output = capsys.readouterr()
    check_progress_bar(output, 5, "build")
    check_progress_bar(output, 5, "prove")


def test_evaluate_directory_no_generations(
    tmp_path: Path,
    expanded_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Test that `evaluate_datasets()` warns when run on base datasets."""
    output_dir = tmp_path / "output"
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD],
            path=expanded_test_datasets,
            output_dir=output_dir,
            jobs=8,
        )
    assert not output_dir.exists()
    for name in ["ada_test", "explain_test", "spark_test"]:
        msg = f"Dataset '{name}' does not contain generations; Skipping evaluation."
        assert_log(caplog, WARN, msg)
    assert_log(caplog, WARN, "No datasets compatible with build found.")
    assert_log(
        caplog, WARN, "No datasets were compatible with any eval; no results to save."
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_evaluate_directory_save_unpacked(
    tmp_path: Path,
    expanded_test_datasets: Path,  # noqa: F811  # pytest fixture
):
    """Test that `evaluate_datasets()` saves in unpacked format when appropriate."""
    # Output should be saved in packed format by default
    output_dir = tmp_path / "output"
    assert not output_dir.exists()
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD],
            path=expanded_test_datasets,
            output_dir=output_dir,
            jobs=8,
            canonical_evaluation=True,
        )
    assert (output_dir / "spark_test.jsonl").is_file()
    assert not (output_dir / "spark_test").exists()
    # If saving to a directory already containing unpacked datasets, output
    # should be saved in unpacked format
    shutil.rmtree(output_dir)
    output_dir.mkdir()
    shutil.copytree(expanded_test_datasets / "ada_test", output_dir / "ada_test")
    with patch("ada_eval.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD],
            path=expanded_test_datasets,
            output_dir=output_dir,
            jobs=8,
            canonical_evaluation=True,
        )
    assert (output_dir / "spark_test").is_dir()
    assert not (output_dir / "spark_test.jsonl").exists()


@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gnatformat"), reason="gnatformat not available")
def test_build(
    eval_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Build eval should support both ada and spark datasets (and treat them
    # identically)
    ada_dataset_file = eval_test_datasets / "ada_build.jsonl"
    spark_dataset_file = eval_test_datasets / "spark_build.jsonl"
    shutil.copy(ada_dataset_file, spark_dataset_file)

    # Apply the build eval to the eval test datasets (for simplicity, they
    # contain initial samples defining only a canonical solution)
    test_datasets = load_datasets(eval_test_datasets)
    test_datasets = evaluate_datasets_canonical([Eval.BUILD], test_datasets, jobs=8)
    assert caplog.text == ""
    check_progress_bar(capsys.readouterr(), 11, "build")  # 2x4 (build) + 3 (prove)

    # Verify that the evaluation results are as expected for the build test
    # datasets
    build_test_datasets = [d for d in test_datasets if d.name == "build"]
    assert len(build_test_datasets) == 2
    for dataset in build_test_datasets:
        samples = {s.name: s for s in dataset.samples}
        assert samples["correct"].canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=True, pre_format_warnings=False, post_format_warnings=False
            )
        ]
        assert samples["unformatted"].canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=True, pre_format_warnings=True, post_format_warnings=False
            )
        ]
        assert samples["warns"].canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=True, pre_format_warnings=True, post_format_warnings=True
            )
        ]
        assert samples["fails"].canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=False, pre_format_warnings=True, post_format_warnings=True
            )
        ]

    # Verify that all the spark samples in the prove dataset compiled without
    # issue.
    prove_test_datasets = [d for d in test_datasets if d.name == "prove"]
    assert len(prove_test_datasets) == 1
    for sample in prove_test_datasets[0].samples:
        assert sample.canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=True, pre_format_warnings=False, post_format_warnings=False
            )
        ]


@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
def test_prove(
    eval_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Apply the prove eval to the eval test datasets (for simplicity, they
    # contain initial samples defining only a canonical solution)
    test_datasets = load_datasets(eval_test_datasets)
    test_datasets = evaluate_datasets_canonical([Eval.PROVE], test_datasets, jobs=8)
    assert caplog.text == ""
    check_progress_bar(capsys.readouterr(), 3, "prove")  # 3 spark samples

    # Verify that the evaluation results are as expected for the build dataset
    # (only spark samples are compatible with prove)
    build_test_datasets = [d for d in test_datasets if d.name == "build"]
    assert len(build_test_datasets) == 1
    for sample in build_test_datasets[0].samples:
        assert sample.canonical_evaluation_results == []

    # Verify that the evaluation results are as expected for the prove dataset
    prove_test_datasets = [d for d in test_datasets if d.name == "prove"]
    assert len(prove_test_datasets) == 1
    samples = {s.name: s for s in prove_test_datasets[0].samples}
    assert samples["proves"].canonical_evaluation_results == [
        EvaluationStatsProve(successfully_proven=True, subprogram_found=True)
    ]
    assert samples["fails"].canonical_evaluation_results == [
        EvaluationStatsProve(successfully_proven=False, subprogram_found=True)
    ]
    assert samples["not_found"].canonical_evaluation_results == [
        EvaluationStatsProve(successfully_proven=False, subprogram_found=False)
    ]


def test_eval_path_checks(eval_test_datasets: Path):  # noqa: F811  # pytest fixture
    """Check that evals raise appropriate exceptions when tools are not available."""
    test_datasets = load_datasets(eval_test_datasets)

    with patch.dict(os.environ, {"PATH": ""}):
        error_msg = "'gprbuild' is not available in the PATH."
        with pytest.raises(ExecutableNotFoundError, match=re.escape(error_msg)):
            evaluate_datasets_canonical([Eval.BUILD], test_datasets, jobs=8)

        error_msg = "'gnatprove' is not available in the PATH."
        with pytest.raises(ExecutableNotFoundError, match=re.escape(error_msg)):
            evaluate_datasets_canonical([Eval.PROVE], test_datasets, jobs=8)
