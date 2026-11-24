import os
import re
import shutil
import subprocess
from logging import ERROR, WARN
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

import pytest
from helpers import assert_git_status, assert_log, dictify_datasets, setup_git_repo

from ada_eval.datasets import Dataset, dataset_has_sample_type
from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.trivial_generations import generate_canonical
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.evaluation_stats import (
    Eval,
    EvaluationStatsBase,
    EvaluationStatsBuild,
    EvaluationStatsFailed,
    EvaluationStatsProve,
    EvaluationStatsTest,
    EvaluationStatsTimedOut,
    ProofCheck,
)
from ada_eval.datasets.types.exit_status import ExitStatus
from ada_eval.datasets.types.metrics import MetricSection, metric_section
from ada_eval.datasets.types.samples import (
    EVALUATED_SAMPLE_TYPES,
    GENERATED_SAMPLE_TYPES,
    EvaluatedAdaSample,
    EvaluatedExplainSample,
    EvaluatedSample,
    EvaluatedSparkSample,
    ExplainSample,
    GeneratedAdaSample,
    GeneratedProofSample,
    GeneratedSample,
    GeneratedSparkSample,
    GenerationStats,
    Location,
    ProofSample,
    Sample,
    SampleKind,
    SparkSample,
)
from ada_eval.evals import (
    evaluate_datasets,
    evaluate_datasets_canonical,
    evaluate_directory,
)
from ada_eval.evals.build import Build
from ada_eval.evals.generic_eval import GenericEval, WrongEvalOutputTypeError
from ada_eval.evals.prove import (
    Prove,
    compile_objects,
    count_pragma_assumes,
    evaluate_proof_harness,
    extract_non_spark_entities,
    extract_proof_results,
    extract_skipped_entities,
    log_gnatprove_failure,
)
from ada_eval.utils import ExecutableNotFoundError, gpr_tools_env

GENERATED_TYPE_TO_EVALUATED: dict[type[GeneratedSample], type[EvaluatedSample]] = {
    GENERATED_SAMPLE_TYPES[k]: EVALUATED_SAMPLE_TYPES[k] for k in SampleKind
}


# Mock eval stats matching those in `evaluated_test_datasets`
MOCK_BUILD_EVAL_STATS = EvaluationStatsBuild(
    compiled=False, pre_format_warnings=True, post_format_warnings=True
)
MOCK_PROVE_EVAL_STATS = EvaluationStatsProve(
    result="unproved",
    proved_checks={"PROVED_CHECK_NAME": 1},
    unproved_checks={"UNPROVED_CHECK_NAME": 2},
    warnings={"WARNING_NAME": 3},
    non_spark_entities=["Entity_Name"],
    missing_required_checks=[
        ProofCheck(
            rule="RULE_NAME",
            entity_name="My_Package.My_Subprogram",
            src_pattern="pattern",
        )
    ],
    pragma_assume_count=4,
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


class MockTestEval(GenericEval[GeneratedSample, EvaluatedSample]):
    """Mock test eval suitable for reproducing `evaluated_test_datasets`."""

    eval: ClassVar = Eval.TEST
    supported_types: ClassVar = GENERATED_TYPE_TO_EVALUATED

    def evaluate(self, _: GeneratedSample) -> EvaluationStatsTest:
        return EvaluationStatsTest(eval=Eval.TEST, compiled=True, passed_tests=True)


def mock_create_eval(eval_: Eval) -> MockBuildEval | MockProveEval | MockTestEval:
    """Mock `create_eval()` suitable for reproducing `evaluated_test_datasets`."""
    match eval_:
        case Eval.BUILD:
            return MockBuildEval()
        case Eval.PROVE:
            return MockProveEval()
        case Eval.TEST:
            return MockTestEval()
        case _:
            raise ValueError(f"Unknown mock eval {eval_}")


def check_progress_bar(output: Any, total: int, eval_name: str):
    assert f"Evaluating with {eval_name}: 100%" in output.err
    assert f" {total}/{total} " in output.err
    assert output.out == ""


def test_generic_eval(
    generated_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Create mock evaluation stats and two evals. Both evals just return the
    # mock stats, but `MockEval1` is incompatible with `ExplainSample`s, raises
    # a subprocess timeout on "test_sample_1", and raises a subprocess error on
    # "test_sample_2".
    class MockEvaluationStats(EvaluationStatsBase):
        passed: ClassVar = True

        def metrics(self, _: EvaluationStatsBase) -> MetricSection:
            return metric_section()

    class MockEval0(GenericEval[GeneratedSample, EvaluatedSample]):
        eval: ClassVar = Eval.BUILD
        supported_types: ClassVar = GENERATED_TYPE_TO_EVALUATED

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
    original_samples = dictify_datasets(generated_datasets)
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
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
                generated_sample = original_samples[(dataset.dirname, sample.name)]
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

    # Test that rerunning the evals on the already-evaluated datasets returns
    # samples with twice as many eval stats, without mutating the existing ones.
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
        doubly_evaluated_datasets = evaluate_datasets(
            evals=[0, 1, 2],  # type: ignore[list-item]  # Using mock IDs instead of real enum
            datasets=evaluated_datasets,
            jobs=8,
        )
    singly_evaluated_samples = dictify_datasets(evaluated_datasets)
    doubly_evaluated_samples = dictify_datasets(doubly_evaluated_datasets)
    for key, singly_evaluated in singly_evaluated_samples.items():
        doubly_evaluated = doubly_evaluated_samples[key]
        assert len(singly_evaluated.evaluation_results) in (1, 2)
        assert list(doubly_evaluated.evaluation_results) == 2 * list(
            singly_evaluated.evaluation_results
        )

    # Run the mock evals as a canonical evaluation on the evaluated datasets
    # (canonical evaluations would usually be run on base datasets, but this
    # should work too, and serves to check that the results of evaluating the
    # generations do not pollute the canonical results).
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
        evaluate_datasets_canonical(
            evals=[2, 1, 0],  # type: ignore[list-item]  # Using mock IDs instead of real enum
            datasets=evaluated_datasets,
            jobs=8,
        )
    # Check that the `MockEvaluationStat`s were merged into the existing
    # `canonical_evaluation_results`, then restore the originals so that
    # `check_evaluated_datasets()` can verify the rest.
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            original_sample = original_samples[(dataset.dirname, sample.name)]
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
    generated_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Test that an exception is raised if `supported_types` is misconfigured."""

    # Create a mock eval with a misconfigured `supported_types` (mapping
    # an `AdaSample` to an `ExplainSample`).
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
        "Eval 'build' purports to map samples of type GeneratedAdaSample to "
        "type EvaluatedExplainSample, but the corresponding evaluated type is "
        "actually EvaluatedAdaSample."
    )
    with (
        patch("ada_eval.evals.evaluate.create_eval", return_value=MockEval()),
        pytest.raises(WrongEvalOutputTypeError, match=re.escape(error_msg)),
    ):
        evaluate_datasets(
            evals=[None],  # type: ignore[list-item]  # Dummy value; just needs length 1
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
    generated_test_datasets: Path, caplog: pytest.LogCaptureFixture
):
    """Test that `evaluate_datasets()` warns when no `Eval`s are provided."""
    with patch("ada_eval.evals.evaluate.create_eval") as mock_create_eval:
        datasets = cast(
            list[Dataset[GeneratedSample]], load_datasets(generated_test_datasets)
        )
        returned_datasets = evaluate_datasets(evals=[], datasets=datasets, jobs=8)
    assert returned_datasets == []
    assert not mock_create_eval.called
    assert_log(caplog, WARN, "No evals provided; skipping evaluation.")


def test_evaluate_directory(
    tmp_path: Path,
    evaluated_test_datasets: Path,
    generated_test_datasets: Path,
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

    # Run `MockBuildEval` and `MockProveEval` on the generated test datasets and
    # check this regenerates the evaluated datasets.
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
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
    original_samples = dictify_datasets(load_datasets(generated_test_datasets))

    # Run the same mock evals as a canonical evaluation on the generated datasets
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
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
            original_sample = original_samples[(dataset.dirname, sample.name)]
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
    expanded_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Test that `evaluate_datasets()` warns when run on initial datasets."""
    output_dir = tmp_path / "output"
    assert not output_dir.exists()
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD],
            path=expanded_test_datasets,
            output_dir=output_dir,
            jobs=8,
        )
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []
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


def test_evaluate_directory_save_unpacked(tmp_path: Path, expanded_test_datasets: Path):
    """Test that `evaluate_datasets()` saves in unpacked format when appropriate."""
    # Output should be saved in packed format by default
    output_dir = tmp_path / "output"
    assert not output_dir.exists()
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
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
    with patch("ada_eval.evals.evaluate.create_eval", mock_create_eval):
        evaluate_directory(
            [Eval.BUILD],
            path=expanded_test_datasets,
            output_dir=output_dir,
            jobs=8,
            canonical_evaluation=True,
        )
    assert (output_dir / "spark_test").is_dir()
    assert not (output_dir / "spark_test.jsonl").exists()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprclean"), reason="gprclean not available")
@pytest.mark.skipif(not shutil.which("gnatformat"), reason="gnatformat not available")
def test_build(
    eval_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Build eval should support both ada and spark datasets (and treat them
    # identically), so make a copy of the ada eval test dataset with the name
    # changed so that it loads as a spark dataset.
    ada_dataset_file = eval_test_datasets / "ada_build.jsonl"
    spark_dataset_file = eval_test_datasets / "spark_build.jsonl"
    shutil.copy(ada_dataset_file, spark_dataset_file)

    # Apply the build eval to the eval test datasets (for simplicity, they
    # contain initial samples defining only a canonical solution)
    test_datasets = load_datasets(eval_test_datasets)
    evaluate_datasets_canonical([Eval.BUILD], test_datasets, jobs=8)
    assert caplog.text == ""
    # Two builds contribute four metrics each, proof fourteen and tests five.
    check_progress_bar(capsys.readouterr(), 27, "build")

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

    # Verify that all the spark samples in the prove dataset (except the
    # deliberately invalid `"errors"`) compiled without issue.
    prove_test_datasets = [d for d in test_datasets if d.name == "prove"]
    assert len(prove_test_datasets) == 1
    for sample in prove_test_datasets[0].samples:
        assert sample.name == "errors" or sample.canonical_evaluation_results == [
            EvaluationStatsBuild(
                compiled=True, pre_format_warnings=False, post_format_warnings=False
            )
        ]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprclean"), reason="gprclean not available")
@pytest.mark.skipif(not shutil.which("gnatformat"), reason="gnatformat not available")
def test_build_checks_only_the_sample_project(eval_test_datasets: Path, tmp_path: Path):
    # This dependency deliberately violates the formatter's style.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_gpr = other_dir / "other.gpr"
    other_gpr.write_text('project Other is\n   for Object_Dir use "obj";\nend Other;\n')
    other_src = other_dir / "other.ads"
    other_text = "package Other is X:constant Integer:=1; end Other;\n"
    other_src.write_text(other_text)

    # Import the fixture library from the unformatted sample.
    dataset = next(d for d in load_datasets(eval_test_datasets) if d.name == "build")
    sample = next(
        s for s in generate_canonical([dataset])[0].samples if s.name == "unformatted"
    )
    assert isinstance(sample, GeneratedAdaSample)
    files = dict(sample.generated_solution.files)
    main_gpr = Path("main.gpr")
    files[main_gpr] = f'with "{other_gpr}";\n'.encode() + files[main_gpr]
    main_src = Path("src/hello.adb")
    files[main_src] = b"with Other;\npragma Unreferenced (Other);\n" + files[main_src]
    sample.generated_solution = DirectoryContents(files)
    sample.canonical_solution = DirectoryContents(files)  # Reference settings.

    # The build should pass after formatting only the sample.
    stats = Build().evaluate(sample)
    assert other_src.read_text() == other_text
    assert stats.compiled
    assert not stats.post_format_warnings


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_prove(
    eval_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Apply the prove eval to the eval test datasets (for simplicity, they
    # contain initial samples defining only a canonical solution)
    all_datasets = load_datasets(eval_test_datasets)
    test_datasets = [d for d in all_datasets if d.name in ("build", "prove")]
    evaluate_datasets_canonical([Eval.PROVE], test_datasets, jobs=8)
    # Expect diagnostics from the two failing GNATprove invocations.
    assert {r.levelno for r in caplog.records} == {WARN}
    assert sorted(r.getMessage().split("\n")[0] for r in caplog.records) == [
        (
            "GNATprove failed with exit code 1 on sample 'changed_profile': "
            "gnatprove -Pharness.gpr --warnings=error -u proof_harness.adb "
            "proof_harness.ads"
        ),
        (
            "GNATprove failed with exit code 1 on sample 'errors': "
            "gnatprove -Pmain.gpr -k increment.ads"
        ),
    ]
    check_progress_bar(capsys.readouterr(), 14, "prove")

    # Verify that the evaluation results are as expected for the build dataset
    # (only spark samples are compatible with prove, so this should be an empty
    # list)
    build_test_datasets = [d for d in test_datasets if d.name == "build"]
    assert len(build_test_datasets) == 1
    for sample in build_test_datasets[0].samples:
        assert sample.canonical_evaluation_results == []

    # Verify that the evaluation results are as expected for the prove dataset
    prove_test_datasets = [d for d in test_datasets if d.name == "prove"]
    assert len(prove_test_datasets) == 1
    samples = {s.name: s for s in prove_test_datasets[0].samples}
    expected_eval_stats = EvaluationStatsProve(
        result="proved",
        proved_checks={
            "SUBPROGRAM_TERMINATION": 1,
            "VC_OVERFLOW_CHECK": 2,
            "VC_POSTCONDITION": 1,
        },
        unproved_checks={},
        warnings={},
        non_spark_entities=[],
        missing_required_checks=[],
        pragma_assume_count=0,
    )
    expected_proof_check = ProofCheck(
        rule="VC_POSTCONDITION",
        entity_name="Increment",
        src_pattern="Increment'Result\\s+=\\s+X\\s+\\+\\s+1;",
    )
    assert samples["proves"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(update={"harness": "proved"})
    ]
    assert samples["child_unit"].canonical_evaluation_results == [expected_eval_stats]
    assert samples["warns"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={"result": "proved_incorrectly", "warnings": {"INEFFECTIVE": 1}}
        )
    ]
    assert samples["fails"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "unproved",
                "proved_checks": {"SUBPROGRAM_TERMINATION": 1, "VC_OVERFLOW_CHECK": 1},
                "unproved_checks": {"VC_OVERFLOW_CHECK": 1, "VC_POSTCONDITION": 1},
                "missing_required_checks": [expected_proof_check],
            }
        )
    ]
    assert samples["assumes"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={"result": "proved_incorrectly", "pragma_assume_count": 1}
        )
    ]
    assert samples["disables_spark_mode"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "proved_incorrectly",
                "proved_checks": {"SUBPROGRAM_TERMINATION": 1},
                "non_spark_entities": ["Increment.Increment"],
            }
        )
    ]
    # An imported stub is a valid exception only when the sample declares it.
    stub_eval_stats = expected_eval_stats.model_copy(
        update={"non_spark_entities": ["Helper"]}
    )
    assert samples["imports_stub"].canonical_evaluation_results == [stub_eval_stats]
    assert samples["undeclared_stub"].canonical_evaluation_results == [
        stub_eval_stats.model_copy(update={"result": "proved_incorrectly"})
    ]
    # The harness catches both a weakened postcondition and a stronger
    # precondition, without relying on the source pattern.
    assert samples["wrong_postcondition"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "proved_incorrectly",
                "missing_required_checks": [expected_proof_check],
                "harness": "unproved",
            }
        )
    ]
    # A proving solution with an incompatible subprogram profile
    assert samples["changed_profile"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={"result": "proved_incorrectly", "harness": "error"}
        )
    ]
    assert samples["no_postcondition"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "proved_incorrectly",
                "proved_checks": {"SUBPROGRAM_TERMINATION": 1, "VC_OVERFLOW_CHECK": 1},
                "missing_required_checks": [
                    expected_proof_check.model_copy(update={"src_pattern": None})
                ],
            }
        )
    ]
    assert samples["delegated_fails"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "unproved",
                "proved_checks": {
                    "SUBPROGRAM_TERMINATION": 2,
                    "VC_OVERFLOW_CHECK": 2,
                    "VC_POSTCONDITION": 1,
                },
                "unproved_checks": {"VC_OVERFLOW_CHECK": 1, "VC_POSTCONDITION": 1},
            }
        )
    ]
    assert samples["errors"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(update={"result": "error", "proved_checks": {}})
    ]
    assert samples["not_found"].canonical_evaluation_results == [
        expected_eval_stats.model_copy(
            update={
                "result": "subprogram_not_found",
                "proved_checks": {},
            }
        )
    ]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_proof_sample_is_built_and_proved_but_not_tested(eval_test_datasets: Path):
    prove_dataset = next(
        d for d in load_datasets(eval_test_datasets) if d.name == "prove"
    )
    spark_sample = next(s for s in prove_dataset.samples if s.name == "proves")
    proof_sample = ProofSample(
        **spark_sample.model_dump(
            exclude={"unit_tests", "canonical_evaluation_results"}
        ),
        canonical_evaluation_results=[],
    )
    dataset = Dataset(name="prove", sample_type=ProofSample, samples=[proof_sample])
    evaluate_datasets_canonical([Eval.BUILD, Eval.PROVE, Eval.TEST], [dataset], jobs=1)
    results = {es.eval: es for es in proof_sample.canonical_evaluation_results}
    assert results.keys() == {Eval.BUILD, Eval.PROVE}
    prove_stats = results[Eval.PROVE]
    assert isinstance(prove_stats, EvaluationStatsProve)
    assert prove_stats.result == "proved"
    assert prove_stats.harness == "proved"


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_prove_whole_project_analyses_a_generic(eval_test_datasets: Path):
    prove_dataset = next(
        d for d in load_datasets(eval_test_datasets) if d.name == "prove"
    )
    spark_sample = next(s for s in prove_dataset.samples if s.name == "proves")
    assert isinstance(spark_sample, SparkSample)
    project_files = {
        path: contents
        for path, contents in spark_sample.canonical_solution.files.items()
        if path.suffix in (".gpr", ".adc")
    }
    # Instantiate a generic with an unsatisfiable postcondition.
    solution = DirectoryContents(
        project_files
        | {
            Path("src/gen.ads"): (
                b"generic\n"
                b"function Gen (X : Integer) return Integer\n"
                b"with Post => Gen'Result = X + 1;\n"
            ),
            Path("src/gen.adb"): (
                b"function Gen (X : Integer) return Integer is\n"
                b"begin\n"
                b"   return X;\n"
                b"end Gen;\n"
            ),
            Path("src/inst.ads"): b"with Gen;\nfunction Inst is new Gen;\n",
        }
    )
    sample = GeneratedProofSample(
        name="generic",
        location=Location(path=Path("src/gen.ads"), subprogram_name="Gen"),
        prompt="",
        comments="",
        sources=solution,
        canonical_solution=solution,
        canonical_evaluation_results=[],
        generation_stats=GenerationStats(
            exit_status=ExitStatus.SUCCESS, stdout="", stderr="", runtime_ms=0
        ),
        generated_solution=solution,
    )
    # Unit-only analysis misses the instance.
    unit_stats = Prove().evaluate(sample)
    assert unit_stats.proved_checks.total() == 0
    assert unit_stats.unproved_checks.total() == 0
    # Project-wide analysis reaches the failing postcondition.
    sample.prove_whole_project = True
    project_stats = Prove().evaluate(sample)
    assert project_stats.result == "unproved"
    assert project_stats.unproved_checks["VC_POSTCONDITION"] == 1


def test_extract_non_spark_entities():
    spark_data: dict[str, object] = {
        "entities": {
            "1": {"name": "P.Own", "sloc": [{"file": "p.adb", "line": 3, "col": 4}]},
            "2": {"name": "P.Lib", "sloc": [{"file": "lib.ads", "line": 1, "col": 1}]},
            "3": {"name": "P.Unlocated"},
            "4": {
                "name": "P.In_Spark",
                "sloc": [{"file": "p.ads", "line": 2, "col": 4}],
            },
            "5": {
                "name": "P.Freegp603.Free",
                "sloc": [{"file": "p.adb", "line": 9, "col": 4}],
            },
            "6": {
                "name": "P.FREE_TEXTGP1096.Free_Text",
                "sloc": [{"file": "p.adb", "line": 12, "col": 4}],
            },
        },
        "spark": {
            "1": "spec",
            "2": "spec",
            "3": "no",
            "4": "all",
            "5": "no",
            "6": "no",
        },
    }
    assert extract_non_spark_entities([spark_data]) == [
        "P.Own",
        "P.Lib",
        "P.Unlocated",
        "P.Free",
        "P.Free_Text",
    ]
    # The source filter retains entities that have no reported location.
    assert extract_non_spark_entities([spark_data], declared_in={"p.ads", "p.adb"}) == [
        "P.Own",
        "P.Unlocated",
        "P.Free",
        "P.Free_Text",
    ]


def test_extract_skipped_entities():
    spark_data: dict[str, object] = {
        "entities": {
            " 1": {"name": "P.Own", "sloc": [{"file": "p.adb", "line": 3}]},
            " 2": {"name": "P.Own.T1B", "sloc": [{"file": "p.adb", "line": 4}]},
            " 3": {"name": "P.Lib", "sloc": [{"file": "lib.adb", "line": 1}]},
            " 4": {"name": "P.Flow", "sloc": [{"file": "p.adb", "line": 9}]},
        },
        "spark": {" 1": "all", " 2": "all", " 3": "all", " 4": "all"},
        "skip_proof": [1, 2, 3],
        "skip_flow_proof": [4],
    }
    # Keep outermost skipped entities from the sample.
    assert extract_skipped_entities([spark_data], declared_in={"p.adb"}) == [
        "P.Flow",
        "P.Own",
    ]
    assert extract_skipped_entities([{"entities": {}, "spark": {}}]) == []


def test_extract_proof_results_marks_justified_checks():
    result = {"entity": 1, "rule": "VC_ASSERT", "severity": "info", "file": "p.adb"}
    spark_data: dict[str, object] = {
        "entities": {" 1": {"name": "P"}},
        "flow": [],
        "proof": [
            result | {"line": 1, "col": 1},
            result | {"line": 2, "col": 1, "suppressed": "trust me"},
        ],
    }
    assert [r.justified for r in extract_proof_results(spark_data)] == [False, True]
    # GNATprove can omit all result sections for a unit.
    assert extract_proof_results({"entities": {}}) == []
    assert count_pragma_assumes([{"entities": {}}]) == 0


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_prove_rejects_justified_checks_and_skipped_proof(eval_test_datasets: Path):
    dataset = next(d for d in load_datasets(eval_test_datasets) if d.name == "prove")
    samples = {s.name: s for s in generate_canonical([dataset])[0].samples}

    def with_file(name: str, path: str, text: str) -> GeneratedSparkSample:
        sample = samples[name]
        assert isinstance(sample, GeneratedSparkSample)
        files = dict(sample.generated_solution.files)
        files[Path(path)] = text.encode()
        return sample.model_copy(
            update={"generated_solution": DirectoryContents(files)}
        )

    # Record the justification on an assertion GNATprove cannot prove.
    justified = Prove().evaluate(
        with_file(
            "proves",
            "src/increment.adb",
            "function Increment (X : Integer) return Integer is\n"
            "begin\n"
            "   pragma Assert (X > 0);\n"
            '   pragma Annotate (GNATprove, False_Positive, "assertion", "trust me");\n'
            "   return X + 1;\n"
            "end Increment;\n",
        )
    )
    assert justified.result == "proved_incorrectly"
    assert justified.justified_checks == {"VC_ASSERT": 1}
    assert "VC_ASSERT" not in justified.proved_checks

    # A skipped body still counts as an incomplete proof.
    skipped = Prove().evaluate(
        with_file(
            "fails",
            "src/increment.ads",
            "function Increment (X : Integer) return Integer\n"
            "with Post => Increment'Result = X + 1,\n"
            "     Annotate => (GNATprove, Skip_Proof);\n",
        )
    )
    assert skipped.result == "proved_incorrectly"
    assert skipped.unproved_checks == {}
    assert skipped.skipped_entities == ["Increment"]

    # Canonical proof switches expose the original failing checks.
    fails = samples["fails"]
    assert isinstance(fails, GeneratedSparkSample)
    gpr = fails.generated_solution.files[Path("main.gpr")].decode()
    limited = with_file(
        "fails",
        "main.gpr",
        gpr.replace(
            "end Main;",
            "   package Prove is\n"
            '      for Proof_Switches ("Ada") use ("--limit-line=increment.adb:1");\n'
            "   end Prove;\nend Main;",
        ),
    )
    assert "--limit-line" in limited.generated_solution.files[Path("main.gpr")].decode()
    limited_stats = Prove().evaluate(limited)
    assert limited_stats.result == "unproved"
    assert limited_stats.unproved_checks != {}
    assert limited_stats.changed_project_files == ["main.gpr"]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_prove_ignores_entity_case_and_supplied_results(eval_test_datasets: Path):
    dataset = next(d for d in load_datasets(eval_test_datasets) if d.name == "prove")
    samples = {
        s.name: s
        for s in generate_canonical([dataset])[0].samples
        if isinstance(s, GeneratedSparkSample)
    }

    def with_files(name: str, files: dict[Path, bytes]) -> GeneratedSparkSample:
        sample = samples[name]
        return sample.model_copy(
            update={
                "generated_solution": DirectoryContents(
                    {**sample.generated_solution.files, **files}
                )
            }
        )

    # Ada identifier case leaves the declared exception valid.
    stub_files = samples["imports_stub"].generated_solution.files
    renamed = Prove().evaluate(
        with_files(
            "imports_stub",
            {
                path: contents.replace(b"Helper", b"HELPER")
                for path, contents in stub_files.items()
                if path.suffix in (".ads", ".adb")
            },
        )
    )
    assert renamed.result == "proved"
    assert renamed.non_spark_entities == ["HELPER"]

    # Collect proof results from the current run only.
    with samples["proves"].generated_solution.unpacked() as working_dir:
        subprocess.run(
            ["gnatprove", "-Pmain.gpr", "-k", "increment.ads"],
            cwd=working_dir,
            check=True,
            capture_output=True,
            env=gpr_tools_env(),
        )
        planted = (working_dir / "obj" / "gnatprove" / "increment.spark").read_bytes()
    with_planted = Prove().evaluate(
        with_files("no_postcondition", {Path("obj/gnatprove/planted.spark"): planted})
    )
    assert with_planted.result == "proved_incorrectly"
    assert with_planted.missing_required_checks != []


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
def test_proof_harness_rejects_assumptions_warnings_and_missing_wrapper_postconditions(
    eval_test_datasets: Path,
):
    dataset = next(d for d in load_datasets(eval_test_datasets) if d.name == "prove")
    sample = next(
        s for s in generate_canonical([dataset])[0].samples if s.name == "proves"
    )
    assert isinstance(sample, GeneratedSparkSample)
    original_files = dict(sample.proof_harness.files)
    body_path = Path("src/proof_harness.adb")

    with sample.generated_solution.unpacked() as solution_dir:
        assert evaluate_proof_harness(sample, solution_dir) == "proved"

        # This makes the postcondition vacuous, but GNATprove still reports it
        # as proved. The pragma_assume entry must invalidate the harness.
        assume_files = dict(original_files)
        assume_files[body_path] = assume_files[body_path].replace(
            b"      return Increment (X);",
            b"      pragma Assume (False);\n      return Increment (X);",
        )
        sample.proof_harness = DirectoryContents(assume_files)
        assert evaluate_proof_harness(sample, solution_dir) == "error"

        # An ineffective initialization is a GNATprove warning, which must also
        # invalidate the harness.
        warning_files = dict(original_files)
        warning_files[body_path] = warning_files[body_path].replace(
            b"   begin\n",
            b"      Copy : Integer := X;\n   begin\n",
        )
        sample.proof_harness = DirectoryContents(warning_files)
        assert evaluate_proof_harness(sample, solution_dir) == "error"

        # Without a postcondition, the wrapper proves but states nothing.
        spec_path = Path("src/proof_harness.ads")
        no_post_files = dict(original_files)
        no_post_files[spec_path] = no_post_files[spec_path].replace(
            b", Post => Check_Increment'Result = X + 1", b""
        )
        sample.proof_harness = DirectoryContents(no_post_files)
        assert evaluate_proof_harness(sample, solution_dir) == "error"

        # A postcondition on an unrelated helper cannot stand in for the
        # wrapper's missing postcondition.
        unrelated_post_files = dict(no_post_files)
        unrelated_post_files[spec_path] = unrelated_post_files[spec_path].replace(
            b"end Proof_Harness;",
            b"   function Dummy (X : Integer) return Integer\n"
            b"   with Post => Dummy'Result = X;\n\nend Proof_Harness;",
        )
        unrelated_post_files[body_path] = unrelated_post_files[body_path].replace(
            b"end Proof_Harness;",
            b"   function Dummy (X : Integer) return Integer is\n"
            b"   begin\n      return X;\n   end Dummy;\n\nend Proof_Harness;",
        )
        sample.proof_harness = DirectoryContents(unrelated_post_files)
        assert evaluate_proof_harness(sample, solution_dir) == "error"

        # GNATprove writes a .spark file for an empty package; it must not
        # count as proof of a property.
        empty_files = dict(original_files)
        empty_files.pop(body_path)
        empty_files[spec_path] = b"package Proof_Harness is end Proof_Harness;\n"
        sample.proof_harness = DirectoryContents(empty_files)
        assert evaluate_proof_harness(sample, solution_dir) == "error"


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprclean"), reason="gprclean not available")
def test_test(
    eval_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Test eval should support both ada and spark datasets (and treat them
    # identically), so make a copy of the spark eval test dataset with the name
    # changed so that it loads as a ada dataset.
    ada_dataset_file = eval_test_datasets / "ada_test.jsonl"
    spark_dataset_file = eval_test_datasets / "spark_test.jsonl"
    shutil.copy(spark_dataset_file, ada_dataset_file)

    # Apply the test eval to the eval test datasets (for simplicity, they
    # contain initial samples defining only a canonical solution)
    all_datasets = load_datasets(eval_test_datasets)
    test_datasets = [d for d in all_datasets if d.name == "test"]
    evaluate_datasets_canonical([Eval.TEST], test_datasets, jobs=8)
    assert caplog.text == ""
    check_progress_bar(capsys.readouterr(), 10, "test")  # 2x5 (test)

    # Verify that the evaluation results are as expected for the test datasets
    assert len(test_datasets) == 2
    for dataset in test_datasets:
        samples = {s.name: s for s in dataset.samples}
        assert samples["catches_missing_assertions"].canonical_evaluation_results == [
            EvaluationStatsTest(compiled=True, passed_tests=False)
        ]
        assert samples["fails_to_build"].canonical_evaluation_results == [
            EvaluationStatsTest(compiled=False, passed_tests=False)
        ]
        assert samples["ignores_warnings"].canonical_evaluation_results == [
            EvaluationStatsTest(compiled=True, passed_tests=True)
        ]
        assert samples["test_fails"].canonical_evaluation_results == [
            EvaluationStatsTest(compiled=True, passed_tests=False)
        ]
        assert samples["passes"].canonical_evaluation_results == [
            EvaluationStatsTest(compiled=True, passed_tests=True)
        ]


def test_eval_path_checks(eval_test_datasets: Path):
    """Check that evals raise appropriate exceptions when tools are not available."""
    test_datasets = load_datasets(eval_test_datasets)

    for eval_, exe in [
        (Eval.BUILD, "gprbuild"),
        (Eval.PROVE, "gnatprove"),
        (Eval.TEST, "gprbuild"),
    ]:
        error_msg = f"'{exe}' is not available in the PATH."
        with (
            patch.dict(os.environ, {"PATH": ""}),
            pytest.raises(ExecutableNotFoundError, match=re.escape(error_msg)),
        ):
            evaluate_datasets_canonical([eval_], test_datasets, jobs=8)


def test_log_gnatprove_failure_shows_the_command_and_the_end_of_the_output(
    caplog: pytest.LogCaptureFixture,
):
    output = "".join(f"line {i}\n" for i in range(100))
    proc_result = subprocess.CompletedProcess(
        ["gnatprove", "-Pmain.gpr", "-k"], 1, stdout=output, stderr="the error\n"
    )
    log_gnatprove_failure("sample", proc_result)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == WARN
    message = caplog.records[0].getMessage()
    assert message.startswith(
        "GNATprove failed with exit code 1 on sample 'sample': "
        "gnatprove -Pmain.gpr -k\n"
    )
    assert message.endswith("line 99\nthe error")
    assert "line 60\n" not in message
    assert "line 61\n" in message


S1_PROJECT = (
    'project Main is\n   for Source_Dirs use ("src");\n   for Object_Dir use "obj";\n'
    '   for Create_Missing_Dirs use "True";\nend Main;\n'
)
S1_SPEC = (
    "package P with SPARK_Mode is\n"
    "   function Count (X : Natural) return Natural with Post => Count'Result = X;\n"
    "end P;\n"
)
S1_BODY = (
    "package body P with SPARK_Mode is\n"
    "   function Count (X : Natural) return Natural is separate;\n"
    "end P;\n"
)
S1_BASE_COUNT = (
    "separate (P)\n"
    "function Count (X : Natural) return Natural is\n"
    "   R : Natural := 0;\n"
    "begin\n"
    "   for I in 1 .. X loop\n"
    "      R := R + 1;\n"
    "   end loop;\n"
    "   return R;\n"
    "end Count;\n"
)
S1_CANONICAL_COUNT = S1_BASE_COUNT.replace(
    "loop\n", "loop\n      pragma Loop_Invariant (R = I - 1);\n", 1
)


def s1_sample(
    changes: dict[str, str] | None = None, base: str = S1_BASE_COUNT
) -> GeneratedProofSample:
    """
    Generate a single-file proof submission from its canonical solution.

    `changes` replaces files of the generated solution (deleting those set to "").
    """

    def contents(count: str) -> DirectoryContents:
        return DirectoryContents(
            {
                Path("main.gpr"): S1_PROJECT.encode(),
                Path("src/p.ads"): S1_SPEC.encode(),
                Path("src/p.adb"): S1_BODY.encode(),
                Path("src/p-count.adb"): count.encode(),
            }
        )

    generated = dict(contents(S1_CANONICAL_COUNT).files)
    for path, text in (changes or {}).items():
        generated[Path(path)] = text.encode()
    return GeneratedProofSample(
        name="s1",
        location=Location(path=Path("src/p.ads"), subprogram_name="Count"),
        prompt="",
        comments="",
        sources=contents(base),
        canonical_solution=contents(S1_CANONICAL_COUNT),
        canonical_evaluation_results=[],
        editable_path=Path("src/p-count.adb"),
        generation_stats=GenerationStats(
            exit_status=ExitStatus.SUCCESS, stdout="", stderr="", runtime_ms=0
        ),
        generated_solution=DirectoryContents(
            {path: text for path, text in generated.items() if text != b""}
        ),
    )


def test_solution_to_evaluate_takes_only_the_editable_file():
    sample = s1_sample(
        {"src/p.ads": "package P is end P;\n", "src/p-count.adb": "-- edited\n"}
    )
    files = sample.solution_to_evaluate().files
    assert files[Path("src/p.ads")] == S1_SPEC.encode()
    assert files[Path("src/p-count.adb")] == b"-- edited\n"
    missing = s1_sample({"src/p-count.adb": ""})
    assert Path("src/p-count.adb") not in missing.solution_to_evaluate().files
    assert missing.solution_to_evaluate().files[Path("src/p.adb")] == S1_BODY.encode()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_prove_compares_the_code_of_the_editable_file():
    # Adding proof annotations preserves the executable code.
    canonical = Prove().evaluate(s1_sample())
    assert canonical.result == "proved"
    assert canonical.changed_object_files == []

    # Replacing the loop can satisfy the contract while changing the object code.
    replaced = Prove().evaluate(
        s1_sample(
            {
                "src/p-count.adb": "separate (P)\n"
                "function Count (X : Natural) return Natural is\n"
                "begin\n   return X;\nend Count;\n"
            }
        )
    )
    assert replaced.result == "proved_incorrectly"
    assert replaced.changed_object_files == ["p.o"]

    # The original postcondition still applies when the submitted spec weakens it.
    weakened = Prove().evaluate(
        s1_sample(
            {
                "src/p.ads": S1_SPEC.replace("Count'Result = X", "Count'Result >= 0"),
                "src/p-count.adb": S1_CANONICAL_COUNT.replace("return R;", "return 0;"),
            }
        )
    )
    assert weakened.result == "unproved"

    # A missing subunit prevents compilation.
    assert Prove().evaluate(s1_sample({"src/p-count.adb": ""})).result == "error"

    # Replacing a base placeholder can produce the canonical object code.
    placeholder = S1_BASE_COUNT.replace("loop\n", "loop\n      null;\n", 1)
    assert compile_objects(s1_sample(base=placeholder).sources) != compile_objects(
        s1_sample().canonical_solution
    )
    kept = S1_CANONICAL_COUNT.replace("loop\n", "loop\n      null;\n", 1)
    for count in (kept, S1_CANONICAL_COUNT):
        stats = Prove().evaluate(
            s1_sample({"src/p-count.adb": count}, base=placeholder)
        )
        assert stats.result == "proved"
        assert stats.changed_object_files == []
