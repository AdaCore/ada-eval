import re
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from unittest.mock import patch

import pytest
from helpers import (
    compacted_test_datasets,  # noqa: F401  # pytest fixture
    generated_test_datasets,  # noqa: F401  # pytest fixture
)

from ada_eval.datasets import Dataset, dataset_has_sample_type
from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types.samples import (
    EvaluatedAdaSample,
    EvaluatedExplainSample,
    EvaluatedSample,
    EvaluatedSparkSample,
    EvaluationStatsBase,
    EvaluationStatsFailed,
    EvaluationStatsTimedOut,
    ExplainSample,
    GeneratedAdaSample,
    GeneratedExplainSample,
    GeneratedSample,
    GeneratedSparkSample,
    Sample,
)
from ada_eval.evals.generic_eval import GenericEval, WrongEvalOutputTypeError
from ada_eval.evaluate import evaluate_datasets, evaluate_datasets_canonical


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
        name: ClassVar[Literal["mock_eval_0"]] = "mock_eval_0"
        supported_types: ClassVar = {
            GeneratedAdaSample: EvaluatedAdaSample,
            GeneratedExplainSample: EvaluatedExplainSample,
            GeneratedSparkSample: EvaluatedSparkSample,
        }

        def evaluate(self, _: GeneratedSample) -> MockEvaluationStats:  # type: ignore[override]  # `MockEvaluationStats` is not a real `EvaluationStats`
            return MockEvaluationStats(eval="mock_eval_0")

    class MockEval1(GenericEval[GeneratedAdaSample, EvaluatedAdaSample]):
        name: ClassVar[Literal["mock_eval_1"]] = "mock_eval_1"
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
            return MockEvaluationStats(eval="mock_eval_1")

    # Also define an eval which is compatible with nothing, to check that it is
    # effectively ignored.
    class MockEval2(GenericEval[GeneratedSample, EvaluatedSample]):
        name: ClassVar[Literal["mock_eval_2"]] = "mock_eval_2"
        supported_types: ClassVar = {}

        def evaluate(self, _):
            return MockEvaluationStats(eval="mock_eval_2")

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
                    eval="mock_eval_1",
                    cmd_timed_out=["cmd", "timeout-arg"],
                    timeout=1.2,
                )
            ]
        if sample.name == "test_sample_2":
            # Raises an exception on samples called "test_sample_2"
            return [
                EvaluationStatsFailed(
                    eval="mock_eval_1",
                    exception="CalledProcessError()",
                )
            ]
        # Otherwise evaluates successfully
        return [MockEvaluationStats(eval="mock_eval_1")]

    def check_evaluated_datasets(datasets: list[Dataset[EvaluatedSample]]):
        for dataset in datasets:
            generated_dataset = next(
                d for d in generated_datasets if d.dirname() == dataset.dirname()
            )
            assert dataset_has_sample_type(dataset, EvaluatedSample)
            for sample in dataset.samples:
                generated_sample = next(
                    s for s in generated_dataset.samples if s.name == sample.name
                )
                assert isinstance(sample, EvaluatedSample)
                assert sample.evaluation_results == [
                    MockEvaluationStats(eval="mock_eval_0"),
                    *mock_1_expected_stats(generated_sample),
                ]
                assert generated_sample.model_dump() == sample.model_dump(
                    exclude={"evaluation_results"}
                )

    # Check that the evaluated samples were created correctly, with suitable
    # progress bars
    check_evaluated_datasets(evaluated_datasets)
    output = capsys.readouterr()
    check_progress_bar(output, 5, "mock_eval_0")  # 5 total samples (1+1+3)
    check_progress_bar(output, 4, "mock_eval_1")  # 4 total samples (1+3)
    assert len(evaluated_datasets) == 3
    assert sum(len(d.samples) for d in evaluated_datasets) == 5

    # The failure should have been logged with a full stack trace, with
    # additional notes containing the stdout and stderr.
    assert "ERROR" in caplog.text
    assert (
        "Error during evaluation of sample test_sample_2\n"
        "Traceback (most recent call last):\n"
    ) in caplog.text
    assert "raise subprocess.CalledProcessError(" in caplog.text
    assert (
        "subprocess.CalledProcessError: "
        "Command '['cmd', 'fail-arg']' returned non-zero exit status 42.\n"
        r"stdout: 'This is a\nmulti-line\nstdout'"
        "\nstderr: 'This is the stderr'"
    ) in caplog.text
    # The timeout should have been logged with a warning but no stack trace
    assert (
        "Evaluation of sample test_sample_1 failed "
        "due to subprocess timeout (1.2 seconds)"
    ) in caplog.text
    assert "test_sample_1\nTraceback" not in caplog.text

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
    # `canonical_evaluation_results` (i.e. appended, since there is no overlap),
    # then remove them so that `check_evaluated_datasets()` can verify the rest.
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            assert isinstance(sample, EvaluatedSample)
            canonical_results: list[EvaluationStatsBase] = list(
                sample.canonical_evaluation_results
            )
            if isinstance(sample, EvaluatedExplainSample):
                # `MockEval1` is incompatible with `ExplainSample`s, so only
                # the results from `MockEval0` should be present
                assert len(canonical_results) >= 1
                assert canonical_results[-1] == MockEvaluationStats(eval="mock_eval_0")
                sample.canonical_evaluation_results.pop()  # Remove `mock_eval_0`
            else:
                assert len(canonical_results) >= 2
                eval_stats_1, eval_stats_0 = canonical_results[-2:]
                assert eval_stats_0 == MockEvaluationStats(eval="mock_eval_0")
                assert [eval_stats_1] == mock_1_expected_stats(sample)
                sample.canonical_evaluation_results.pop()  # Remove `mock_eval_0`
                sample.canonical_evaluation_results.pop()  # Remove `mock_eval_1`
    # Check that everything else is still correct
    check_evaluated_datasets(evaluated_datasets)
    assert len(evaluated_datasets) == 3
    assert sum(len(d.samples) for d in evaluated_datasets) == 5


def test_generic_eval_wrong_output_type(
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    """Test that an exception is raised if `supported_types` is misconfigured."""

    # Create a mock eval with a misconfigured `supported_types` (mapping
    # `AdaSample` to `ExplainSample`).
    class MockEval(GenericEval[GeneratedSample, EvaluatedSample]):
        name: ClassVar[Literal["mock_eval"]] = "mock_eval"
        supported_types: ClassVar = {
            GeneratedAdaSample: EvaluatedExplainSample,
            GeneratedSparkSample: EvaluatedSparkSample,
        }

        def evaluate(self, _: GeneratedSample) -> EvaluationStatsFailed:
            # Use EvaluationStatsFailed as a dummy return value
            return EvaluationStatsFailed(eval="mock_eval", exception="")

    # Check that running this eval on datasets including an `AdaSample` raises
    # a `WrongEvalOutputTypeError`.
    error_msg = (
        "Eval 'mock_eval' accepted a GeneratedSample of type GeneratedAdaSample, "
        "but the corresponding evaluated sample type (EvaluatedAdaSample) is "
        "not compatible with the eval's output types "
        "(EvaluatedExplainSample, EvaluatedSparkSample)."
    )
    with (
        pytest.raises(WrongEvalOutputTypeError, match=re.escape(error_msg)),
        patch("ada_eval.evaluate.create_eval", return_value=MockEval()),
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
    assert "Evaluating with mock_eval:" in output.err
