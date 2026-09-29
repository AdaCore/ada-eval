import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ada_eval.cli import main
from ada_eval.datasets.types import ExitStatus, GenerationStats
from ada_eval.datasets.types.execution_log import ReasoningLogEntry
from ada_eval.paths import (
    EXPANDED_DATASETS_DIR,
    PROJECT_ROOT,
    SMOKE_TEST_DIR,
    TOOL_CONFIGS_DIR,
)
from ada_eval.smoke_test import (
    DEFAULT_SMOKE_SAMPLE,
    REQUIRED_EXECUTABLES,
    default_output_dir,
    last_output_line,
    run_smoke_test,
)

DUMMY_CONFIG = PROJECT_ROOT / "tools" / "configs" / "shell_dummy.json"


def test_default_sample_exists():
    assert (EXPANDED_DATASETS_DIR / DEFAULT_SMOKE_SAMPLE).is_dir()


def test_default_output_dir(tmp_path: Path):
    assert (
        default_output_dir(TOOL_CONFIGS_DIR / "opencode" / "model.json")
        == SMOKE_TEST_DIR / "opencode" / "model"
    )
    assert default_output_dir(tmp_path / "model.json") == SMOKE_TEST_DIR / "model"


def _sample_with_output(
    stdout: str = "", stderr: str = "", reasoning: tuple[str, ...] = ()
) -> MagicMock:
    sample = MagicMock()
    sample.generation_stats = GenerationStats(
        exit_status=ExitStatus.PROCESS_ERROR,
        stdout=stdout,
        stderr=stderr,
        runtime_ms=0,
        execution_log=[ReasoningLogEntry(timestamp_ms=0, text=t) for t in reasoning],
    )
    return sample


def test_last_output_line():
    assert last_output_line(_sample_with_output()) is None
    assert last_output_line(_sample_with_output(stdout="a\nb\n")) == "b"
    assert last_output_line(_sample_with_output("out", "err1\nerr2")) == "err2"
    assert (
        last_output_line(_sample_with_output("out", "err", ("first", "API Error")))
        == "API Error"
    )
    assert last_output_line(_sample_with_output("x" * 10), max_length=4) == "xxxx..."


def test_missing_config_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output_dir = tmp_path / "out"
    passed = run_smoke_test(
        tool_config=tmp_path / "missing.json",
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=output_dir,
    )
    assert not passed
    out = capsys.readouterr().out
    assert "✗ Load tool config" in out
    assert "FileNotFoundError" in out
    assert "Find Ada/SPARK tools" not in out
    assert "Smoke test FAILED" in out
    assert (output_dir / ".gitignore").read_text() == "*\n"


def test_missing_executable_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    def which(name: str) -> str | None:
        return None if name == "gnatprove" else f"/usr/bin/{name}"

    with patch("ada_eval.smoke_test.shutil.which", which):
        passed = run_smoke_test(
            tool_config=DUMMY_CONFIG,
            datasets_dir=EXPANDED_DATASETS_DIR,
            output_dir=tmp_path,
        )
    assert not passed
    out = capsys.readouterr().out
    assert "✓ Load tool config" in out
    assert "✗ Find Ada/SPARK tools" in out
    assert "not on PATH: gnatprove" in out


def test_missing_sample_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with patch("ada_eval.smoke_test.shutil.which", return_value="/usr/bin/x"):
        passed = run_smoke_test(
            tool_config=DUMMY_CONFIG,
            datasets_dir=EXPANDED_DATASETS_DIR,
            output_dir=tmp_path,
            sample_id="spark_learn/no_such_sample",
        )
    assert not passed
    assert "✗ Load sample" in capsys.readouterr().out


def test_existing_output_is_replaced(tmp_path: Path):
    (tmp_path / "stale.txt").write_text("old results")
    run_smoke_test(
        tool_config=tmp_path / "missing.json",
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    assert not (tmp_path / "stale.txt").exists()


def test_cli(tmp_path: Path):
    config = tmp_path / "my_config.json"
    args = ["ada-eval", "smoke-test", "--tool-config", str(config)]
    with (
        patch.object(sys, "argv", args),
        patch("ada_eval.cli.run_smoke_test", return_value=True) as mock_run,
    ):
        main()
    mock_run.assert_called_once_with(
        tool_config=config,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=SMOKE_TEST_DIR / "my_config",
        sample_id=DEFAULT_SMOKE_SAMPLE,
        allow_unsolved=False,
    )

    args += ["--sample", "a/b", "--output-dir", str(tmp_path), "--allow-unsolved"]
    with (
        patch.object(sys, "argv", args),
        patch("ada_eval.cli.run_smoke_test", return_value=False) as mock_run,
        pytest.raises(SystemExit) as exit_info,
    ):
        main()
    assert exit_info.value.code == 1
    assert mock_run.call_args.kwargs["sample_id"] == "a/b"
    assert mock_run.call_args.kwargs["output_dir"] == tmp_path
    assert mock_run.call_args.kwargs["allow_unsolved"]


@pytest.mark.slow
@pytest.mark.skipif(
    not all(shutil.which(e) for e in REQUIRED_EXECUTABLES),
    reason="Ada/SPARK toolchain not available",
)
def test_dummy_tool_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
        allow_unsolved=True,
    )
    out = capsys.readouterr().out
    assert "✓ Evaluate canonical solution" in out
    assert "✓ Evaluate generation" in out
    assert (tmp_path / "generated").is_dir()
    assert (tmp_path / "evaluated").is_dir()

    assert not run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "✗ Evaluate generation" in out
    assert "the sample was not solved" in out
