import re
import shutil
import textwrap
from logging import WARN
from pathlib import Path

import pytest
from helpers import assert_log

from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types.datasets import save_datasets
from ada_eval.datasets.types.samples import MissingCanonicalEvalResultsError
from ada_eval.report import ReportCLIArgs, report_evaluation_results


def test_report_evaluation_results(
    report_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    def test_output(args: ReportCLIArgs, expected_output: str) -> None:
        report_evaluation_results(args)
        output = capsys.readouterr()
        assert output.err == ""
        assert output.out == expected_output

    # Check the report for all samples.
    expected_full_output = textwrap.dedent(
        """\
        total samples:                      10 samples
            passed all evaluations:             2 samples (20.00%)
            generation runtime / s:             5.947 (min 0.0; max 1.0; mean 0.595)
            generation exit code non-zero:      1 sample (10.00%)

        build:                              9 samples (90.00%)
            compiled:                           6 samples (66.67%)
                no warnings:                        2 samples (33.33%)
                formatting warnings:                1 sample (16.67%)
                other warnings:                     3 samples (50.00%)
            failed to compile:                  2 samples (22.22%)
            evaluation errors:                  1 sample (11.11%)

        prove:                              9 samples (90.00%)
            proved correctly:                   1 sample (11.11%)
                proof steps:                        1 (min 1; max 1; mean 1)
                absent checks:                      2 (1 sample; 100.00%)
                unnecessary checks:                 3 (1 sample; 100.00%)
            proved incorrectly:                 3 samples (33.33%)
                missing required checks:            2 (2 samples; 66.67%)
                non-spark entities:                 3 (2 samples; 66.67%)
                pragma assume:                      9 (2 samples; 66.67%)
                warnings:                           4 (2 samples; 66.67%)
            unproved:                           2 samples (22.22%)
            error:                              1 sample (11.11%)
            subprogram not found:               1 sample (11.11%)
            evaluation errors:                  1 sample (11.11%)

        test:                               9 samples (90.00%)
            passed:                             2 samples (22.22%)
            tests failed:                       3 samples (33.33%)
            compilation failed:                 3 samples (33.33%)
            evaluation errors:                  1 sample (11.11%)
        """
    )
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds=None,
            samples=None,
            with_metric=None,
        ),
        expected_full_output,
    )

    # Duplicate the dataset (with kind `Ada`) and check that the results are
    # appropriately doubled.
    shutil.copy(
        report_test_datasets / "spark_report.jsonl",
        report_test_datasets / "ada_report.jsonl",
    )
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds=None,
            samples=None,
            with_metric=None,
        ),
        textwrap.dedent(
            """\
            total samples:                      20 samples
                passed all evaluations:             4 samples (20.00%)
                generation runtime / s:             11.894 (min 0.0; max 1.0; mean 0.595)
                generation exit code non-zero:      2 samples (10.00%)

            build:                              18 samples (90.00%)
                compiled:                           12 samples (66.67%)
                    no warnings:                        4 samples (33.33%)
                    formatting warnings:                2 samples (16.67%)
                    other warnings:                     6 samples (50.00%)
                failed to compile:                  4 samples (22.22%)
                evaluation errors:                  2 samples (11.11%)

            prove:                              18 samples (90.00%)
                proved correctly:                   2 samples (11.11%)
                    proof steps:                        2 (min 1; max 1; mean 1)
                    absent checks:                      4 (2 samples; 100.00%)
                    unnecessary checks:                 6 (2 samples; 100.00%)
                proved incorrectly:                 6 samples (33.33%)
                    missing required checks:            4 (4 samples; 66.67%)
                    non-spark entities:                 6 (4 samples; 66.67%)
                    pragma assume:                      18 (4 samples; 66.67%)
                    warnings:                           8 (4 samples; 66.67%)
                unproved:                           4 samples (22.22%)
                error:                              2 samples (11.11%)
                subprogram not found:               2 samples (11.11%)
                evaluation errors:                  2 samples (11.11%)

            test:                               18 samples (90.00%)
                passed:                             4 samples (22.22%)
                tests failed:                       6 samples (33.33%)
                compilation failed:                 6 samples (33.33%)
                evaluation errors:                  2 samples (11.11%)
            """  # noqa: E501
        ),
    )

    # Check that filtering down to only one dataset (both by name and by kind)
    # gives the original, un-doubled results.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets={"ada_report"},
            dataset_kinds=None,
            samples=None,
            with_metric=None,
        ),
        expected_full_output,
    )
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds={"spark"},
            samples=None,
            with_metric=None,
        ),
        expected_full_output,
    )

    # Test filtering by sample name, and that metrics which apply to no samples
    # are omitted.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds={"spark"},
            samples={"passed", "uncompilable"},
            with_metric=None,
        ),
        textwrap.dedent(
            """\
            total samples:               2 samples
                passed all evaluations:      1 sample (50.00%)
                generation runtime / s:      1.123 (min 0.123; max 1.0; mean 0.561)

            build:                       2 samples (100.00%)
                compiled:                    1 sample (50.00%)
                    no warnings:                 1 sample (100.00%)
                failed to compile:           1 sample (50.00%)

            prove:                       2 samples (100.00%)
                proved correctly:            1 sample (50.00%)
                    proof steps:                 1 (min 1; max 1; mean 1)
                    absent checks:               2 (1 sample; 100.00%)
                    unnecessary checks:          3 (1 sample; 100.00%)
                error:                       1 sample (50.00%)

            test:                        2 samples (100.00%)
                passed:                      1 sample (50.00%)
                compilation failed:          1 sample (50.00%)
            """
        ),
    )

    # Test filtering by presence of a metric.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds={"spark"},
            samples=None,
            with_metric=[["prove", "proved incorrectly", "warnings"]],
        ),
        textwrap.dedent(
            """\
            total samples:                    2 samples
                generation runtime / s:           0.579 (min 0.123; max 0.456; mean 0.289)

            build:                            2 samples (100.00%)
                compiled:                         2 samples (100.00%)
                    formatting warnings:              1 sample (50.00%)
                    other warnings:                   1 sample (50.00%)

            prove:                            2 samples (100.00%)
                proved incorrectly:               2 samples (100.00%)
                    missing required checks:          2 (2 samples; 100.00%)
                    non-spark entities:               1 (1 sample; 50.00%)
                    pragma assume:                    4 (1 sample; 50.00%)
                    warnings:                         4 (2 samples; 100.00%)

            test:                             2 samples (100.00%)
                tests failed:                     1 sample (50.00%)
                compilation failed:               1 sample (50.00%)
            """  # noqa: E501
        ),
    )

    # Test filtering by presence of multiple metrics.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds={"spark"},
            samples=None,
            with_metric=[
                ["prove", "proved incorrectly", "warnings"],
                ["test", "compilation failed"],
            ],
        ),
        textwrap.dedent(
            """\
            total samples:                    1 sample
                generation runtime / s:           0.456 (min 0.456; max 0.456; mean 0.456)

            build:                            1 sample (100.00%)
                compiled:                         1 sample (100.00%)
                    formatting warnings:              1 sample (100.00%)

            prove:                            1 sample (100.00%)
                proved incorrectly:               1 sample (100.00%)
                    missing required checks:          1 (1 sample; 100.00%)
                    warnings:                         3 (1 sample; 100.00%)

            test:                             1 sample (100.00%)
                compilation failed:               1 sample (100.00%)
            """  # noqa: E501
        ),
    )

    # Test samples with no evaluation results.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds={"spark"},
            samples={"no_eval_results"},
            with_metric=None,
        ),
        textwrap.dedent(
            """\
            total samples:               1 sample
                passed all evaluations:      1 sample (100.00%)
                generation runtime / s:      1 (min 1.0; max 1.0; mean 1)
            """
        ),
    )

    # Check that there is meaningful output when no samples match the filters.
    for list_mode in (False, True):
        test_output(
            ReportCLIArgs(
                dataset_dirs=[report_test_datasets],
                datasets={"nonexistent_dataset"},
                dataset_kinds=None,
                samples=None,
                with_metric=None,
                list_samples=list_mode,
            ),
            "No samples matched the specified filters.\n",
        )

    # Test listing selected samples.
    test_output(
        ReportCLIArgs(
            dataset_dirs=[report_test_datasets],
            datasets=None,
            dataset_kinds=None,
            samples=None,
            with_metric=[["build", "compiled"]],
            list_samples=True,
        ),
        textwrap.dedent(
            """\
            ada_report
                passed
                proved_incorrectly_0
                proved_incorrectly_1
                proved_incorrectly_2
                unproved
                unproved_generation_fail
            spark_report
                passed
                proved_incorrectly_0
                proved_incorrectly_1
                proved_incorrectly_2
                unproved
                unproved_generation_fail
            """
        ),
    )

    # Nothing should have been logged at any point.
    assert len(caplog.records) == 0


def test_report_evaluation_results_missing_canonical(report_test_datasets: Path):
    """Check that an evaluation result missing a canonical result is detected."""
    dataset = load_datasets(report_test_datasets / "spark_report.jsonl")[0]
    passed_sample = next(s for s in dataset.samples if s.name == "passed")
    passed_sample.canonical_evaluation_results = (
        passed_sample.canonical_evaluation_results[:1]
    )
    save_datasets([dataset], report_test_datasets)
    error_msg = (
        "sample 'passed' is missing canonical evaluation results for evals "
        "['prove', 'test']."
    )
    with pytest.raises(MissingCanonicalEvalResultsError, match=re.escape(error_msg)):
        report_evaluation_results(
            ReportCLIArgs(
                dataset_dirs=[report_test_datasets],
                datasets=None,
                dataset_kinds=None,
                samples=None,
                with_metric=None,
            )
        )


def test_report_evaluation_results_wrong_dataset_type(
    compacted_test_datasets: Path,
    report_test_datasets: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    """Check that a dataset with the wrong sample type is skipped with a warning."""

    def check_log() -> None:
        for dirname in ("ada_test", "explain_test", "spark_test"):
            skip_msg = (
                f"Skipping dataset '{dirname}' as it does not contain "
                "evaluated samples."
            )
            assert_log(caplog, WARN, skip_msg)
        assert len(caplog.records) == 3
        caplog.clear()

    report_evaluation_results(
        ReportCLIArgs(
            dataset_dirs=[compacted_test_datasets],
            datasets=None,
            dataset_kinds=None,
            samples=None,
            with_metric=None,
        )
    )
    check_log()
    output = capsys.readouterr()
    assert output.out == "No samples matched the specified filters.\n"
    assert output.err == ""

    report_evaluation_results(
        ReportCLIArgs(
            dataset_dirs=[compacted_test_datasets, report_test_datasets],
            datasets=None,
            dataset_kinds=None,
            samples=None,
            with_metric=None,
        )
    )
    check_log()
    output = capsys.readouterr()
    assert output.out.startswith("total samples:")
    assert output.err == ""
