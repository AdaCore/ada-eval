import itertools
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ada_eval.cli import main
from ada_eval.datasets import Eval
from ada_eval.paths import (
    COMPACTED_DATASETS_DIR,
    EVALUATED_DATASETS_DIR,
    EXPANDED_DATASETS_DIR,
    GENERATED_DATASETS_DIR,
)
from ada_eval.tools.factory import Tool


def test_no_args(capsys):
    with patch.object(sys, "argv", ["ada-eval"]), pytest.raises(SystemExit):
        main()
    output = capsys.readouterr()
    assert "error: the following arguments are required: {" in output.err
    assert output.out == ""


def test_generate(capsys):
    # Helper function to patch `sys.argv`
    def patch_args(
        tool: str | None = None,
        tool_config_file: str | None = None,
        dataset: str | None = None,
        jobs: str | None = None,
    ):
        test_args = ["ada-eval", "generate"]
        if tool is not None:
            test_args += ["--tool", tool]
        if tool_config_file is not None:
            test_args += ["--tool-config-file", tool_config_file]
        if dataset is not None:
            test_args += ["--dataset", dataset]
        if jobs is not None:
            test_args += ["--jobs", jobs]
        return patch.object(sys, "argv", test_args)

    # Mock the tool factory
    mock_tool = Mock()
    mock_create_tool = Mock(return_value=mock_tool)
    with patch("ada_eval.cli.create_tool", mock_create_tool):
        # Test with no arguments (should complain about missing `--tool` and
        # `--tool-config-file`)
        with patch_args(), pytest.raises(SystemExit):
            main()
        output = capsys.readouterr()
        assert (
            "error: the following arguments are required: --tool, --tool-config-file"
            in output.err
        )
        assert output.out == ""
        mock_create_tool.assert_not_called()
        mock_tool.apply_to_directory.assert_not_called()

        # Test with an invalid tool name
        with patch_args("invalid_tool", "path/to/config"), pytest.raises(SystemExit):
            main()
        output = capsys.readouterr()
        assert "argument --tool: invalid Tool value: 'invalid_tool'" in output.err
        assert output.out == ""
        mock_create_tool.assert_not_called()
        mock_tool.apply_to_directory.assert_not_called()

        # Test with various valid argument combinations
        mock_cpu_count = Mock(return_value=8)
        cpu_count_patch = patch("ada_eval.cli.cpu_count", mock_cpu_count)
        for tool, dataset, jobs in itertools.product(
            ["shell_script", "SHELL_SCRIPT", "ShElL_ScRiPt"],
            [None, "path/to/dataset"],
            [None, "2", "4"],
        ):
            with patch_args(tool, "path/to/config", dataset, jobs), cpu_count_patch:
                main()
            output = capsys.readouterr()
            assert output.err == ""
            assert output.out == ""
            mock_create_tool.assert_called_once_with(
                Tool.SHELL_SCRIPT, Path("path/to/config")
            )
            dataset_path = COMPACTED_DATASETS_DIR if dataset is None else Path(dataset)
            mock_tool.apply_to_directory.assert_called_once_with(
                path=dataset_path,
                output_dir=GENERATED_DATASETS_DIR,
                jobs=8 if jobs is None else int(jobs),
            )
            mock_create_tool.reset_mock()


def test_evaluate(capsys):
    # Helper function to patch `sys.argv`
    def patch_args(
        evals: list[str] | None = None,
        dataset: str | None = None,
        jobs: str | None = None,
        *,
        canonical: bool = False,
    ):
        test_args = ["ada-eval", "evaluate"]
        if canonical:
            test_args.append("--canonical")
        if evals is not None:
            test_args += ["--evals", *evals]
        if dataset is not None:
            test_args += ["--dataset", dataset]
        if jobs is not None:
            test_args += ["--jobs", jobs]
        return patch.object(sys, "argv", test_args)

    # Mock the `evaluate_directory()` and `cpu_count()` functions
    mock_evaluate_directory = Mock()
    mock_cpu_count = Mock(return_value=8)
    eval_dir_patch = patch("ada_eval.cli.evaluate_directory", mock_evaluate_directory)
    cpu_count_patch = patch("ada_eval.cli.cpu_count", mock_cpu_count)
    with eval_dir_patch, cpu_count_patch:
        # Test with an invalid eval name
        with patch_args(["invalid_eval"]), pytest.raises(SystemExit):
            main()
        output = capsys.readouterr()
        assert "argument --evals: invalid Eval value: 'invalid_eval'" in output.err
        assert output.out == ""
        mock_evaluate_directory.assert_not_called()

        # Test with various valid argument combinations
        for evals, dataset, jobs, canonical in itertools.product(
            [None, ["PrOvE", "build"], ["prove"]],
            [None, "path/to/dataset"],
            [None, "2", "4"],
            [False, True],
        ):
            with patch_args(evals, dataset, jobs, canonical=canonical):
                main()
            output = capsys.readouterr()
            assert output.err == ""
            assert output.out == ""
            expected_evals = (
                [Eval.BUILD, Eval.PROVE, Eval.TEST]
                if evals is None
                else ([Eval.PROVE, Eval.BUILD] if "build" in evals else [Eval.PROVE])
            )
            if canonical:
                expected_dataset_path = (
                    EXPANDED_DATASETS_DIR if dataset is None else Path(dataset)
                )
                expected_output_dir = expected_dataset_path
            else:
                expected_dataset_path = (
                    GENERATED_DATASETS_DIR if dataset is None else Path(dataset)
                )
                expected_output_dir = EVALUATED_DATASETS_DIR
            mock_evaluate_directory.assert_called_once_with(
                evals=expected_evals,
                path=expected_dataset_path,
                output_dir=expected_output_dir,
                jobs=8 if jobs is None else int(jobs),
                canonical_evaluation=canonical,
            )
            mock_evaluate_directory.reset_mock()


def test_check_datasets(capsys):
    # Helper function to patch `sys.argv`
    def patch_args(datasets: list[str] | None = None, jobs: str | None = None):
        test_args = ["ada-eval", "check-datasets"]
        if datasets is not None:
            test_args += ["--datasets", *datasets]
        if jobs is not None:
            test_args += ["--jobs", jobs]
        return patch.object(sys, "argv", test_args)

    # Mock the `check_base_datasets()` and `cpu_count()` functions
    mock_check_base_datasets = Mock()
    mock_cpu_count = Mock(return_value=8)
    check_base_datasets_patch = patch(
        "ada_eval.cli.check_base_datasets", mock_check_base_datasets
    )
    cpu_count_patch = patch("ada_eval.cli.cpu_count", mock_cpu_count)
    with check_base_datasets_patch, cpu_count_patch:
        # Test with empty `--datasets`
        with patch_args([]), pytest.raises(SystemExit):
            main()
        output = capsys.readouterr()
        assert "argument --datasets: expected at least one argument" in output.err
        assert output.out == ""
        mock_check_base_datasets.assert_not_called()

        # Test with various valid argument combinations
        for datasets, jobs in itertools.product(
            [None, ["path/to/dataset"], ["path/to/dataset1", "path/to/dataset2"]],
            [None, "2", "4"],
        ):
            with patch_args(datasets, jobs):
                main()
            output = capsys.readouterr()
            assert output.err == ""
            assert output.out == ""
            mock_check_base_datasets.assert_called_once_with(
                dataset_dirs=(
                    [EXPANDED_DATASETS_DIR, COMPACTED_DATASETS_DIR]
                    if datasets is None
                    else [Path(d) for d in datasets]
                ),
                jobs=8 if jobs is None else int(jobs),
            )
            mock_check_base_datasets.reset_mock()


def test_report(capsys: pytest.CaptureFixture[str]):
    # Helper function to patch `sys.argv`
    def patch_args(  # noqa: PLR0913
        dataset_dirs: list[str] | None = None,
        datasets: set[str] | None = None,
        dataset_kinds: set[str] | None = None,
        samples: set[str] | None = None,
        with_metric: list[list[str]] | None = None,
        *,
        list_samples: bool = False,
    ):
        test_args = ["ada-eval", "report"]
        if dataset_dirs is not None:
            test_args += ["--dataset-dirs", *dataset_dirs]
        if datasets is not None:
            test_args += ["--datasets", *datasets]
        if dataset_kinds is not None:
            test_args += ["--dataset-kinds", *dataset_kinds]
        if samples is not None:
            test_args += ["--samples", *samples]
        if with_metric is not None:
            for metric_path in with_metric:
                test_args += ["--with-metric", *metric_path]
        if list_samples:
            test_args.append("--list-samples")
        return patch.object(sys, "argv", test_args)

    # Mock the `report_evaluation_results()` function
    mock_report_evaluation_results = Mock()
    report_patch = patch(
        "ada_eval.cli.report_evaluation_results", mock_report_evaluation_results
    )
    with report_patch:
        # Test with an invalid dataset kind
        with patch_args(dataset_kinds={"invalid"}), pytest.raises(SystemExit):
            main()
        output = capsys.readouterr()
        assert (
            "argument --dataset-kinds: invalid SampleKind value: 'invalid'"
            in output.err
        )
        assert output.out == ""
        mock_report_evaluation_results.assert_not_called()

        # Test with various valid argument combinations
        for (
            dataset_dirs,
            datasets,
            dataset_kinds,
            samples,
            with_metric,
            list_samples,
        ) in itertools.product(
            [None, ["path/to/dataset_dir"], ["dir1", "dir2"]],
            [None, {"dataset1"}, {"dataset1", "dataset2"}],
            [None, {"ada"}, {"EXPLAIN", "sPaRk"}],
            [None, {"sample1"}, {"sample1", "sample2"}],
            [None, [["metric0"]], [["metric1", "submetric0"], ["metric2"]]],
            [False, True],
        ):
            with patch_args(
                dataset_dirs=dataset_dirs,
                datasets=datasets,
                dataset_kinds=dataset_kinds,
                samples=samples,
                with_metric=with_metric,
                list_samples=list_samples,
            ):
                main()
            output = capsys.readouterr()
            assert output.err == ""
            assert output.out == ""
            expected_dataset_dirs = (
                [EVALUATED_DATASETS_DIR]
                if dataset_dirs is None
                else [Path(d) for d in dataset_dirs]
            )
            expected_dataset_kinds = (
                None
                if dataset_kinds is None
                else {kind.lower() for kind in dataset_kinds}
            )
            mock_report_evaluation_results.assert_called_once_with(
                dataset_dirs=expected_dataset_dirs,
                datasets_filter=datasets,
                dataset_kinds_filter=expected_dataset_kinds,
                samples_filter=samples,
                metrics_filter=with_metric,
                list_samples=list_samples,
            )
            mock_report_evaluation_results.reset_mock()
