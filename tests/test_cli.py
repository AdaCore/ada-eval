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
