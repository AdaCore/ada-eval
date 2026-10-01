import shutil
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ada_eval.cli import main
from ada_eval.datasets.trivial_generations import generate_canonical
from ada_eval.datasets.types import (
    EVALUATED_SAMPLE_TYPES,
    Dataset,
    Eval,
    EvaluationStats,
    EvaluationStatsBuild,
    EvaluationStatsFailed,
    EvaluationStatsTimedOut,
    ExitStatus,
    GenerationStats,
)
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
    load_sample,
    run_smoke_test,
)

DUMMY_CONFIG = PROJECT_ROOT / "tools" / "configs" / "shell_dummy.json"


@pytest.fixture
def smoke_evaluator() -> Iterator[MagicMock]:
    (generated,) = generate_canonical(
        [load_sample(EXPANDED_DATASETS_DIR, DEFAULT_SMOKE_SAMPLE)]
    )
    sample = generated.samples[0]
    evaluated_type = EVALUATED_SAMPLE_TYPES[generated.kind]
    evaluated = Dataset(
        name=generated.name,
        sample_type=evaluated_type,
        samples=[
            evaluated_type(
                **sample.model_dump(),
                evaluation_results=[
                    EvaluationStatsBuild(
                        compiled=True,
                        pre_format_warnings=False,
                        post_format_warnings=False,
                    )
                ],
            )
        ],
    )
    tool = MagicMock()
    tool.name = "dummy"
    tool.config.timeout_s = 1
    tool.apply.return_value = sample
    with (
        patch("ada_eval.smoke_test.create_tool_from_config", return_value=tool),
        patch("ada_eval.smoke_test.shutil.which", return_value="/usr/bin/x"),
        patch(
            "ada_eval.smoke_test.evaluate_datasets", return_value=[evaluated]
        ) as evaluator,
    ):
        yield evaluator


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
    assert not output_dir.exists()
    assert "results in" not in out


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
    assert not list(tmp_path.iterdir())


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
    assert not list(tmp_path.iterdir())


def test_invalid_config_preserves_existing_output(tmp_path: Path):
    (tmp_path / "stale.txt").write_text("old results")
    run_smoke_test(
        tool_config=tmp_path / "missing.json",
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    assert (tmp_path / "stale.txt").read_text() == "old results"
    assert list(tmp_path.iterdir()) == [tmp_path / "stale.txt"]


def test_runs_replace_existing_results(
    tmp_path: Path, smoke_evaluator: MagicMock, capsys: pytest.CaptureFixture[str]
):
    assert run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    for directory in ("generated", "evaluated"):
        (dataset_file,) = (tmp_path / directory).glob("*.jsonl")
        shutil.copyfile(dataset_file, tmp_path / directory / "spark_stale.jsonl")
    assert run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    for directory in ("generated", "evaluated"):
        assert not (tmp_path / directory / "spark_stale.jsonl").exists()
        assert list((tmp_path / directory).glob("*.jsonl"))
    assert (tmp_path / ".gitignore").read_text() == "*\n"
    assert f"results in '{tmp_path}'" in capsys.readouterr().out
    assert smoke_evaluator.call_count == 4


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    [
        ("notes.txt", "unrelated file"),
        ("generated/notes.txt", "unrelated file"),
        ("generated/nested/notes.txt", "unrelated file"),
        ("generated/spark_bad.jsonl", "{}"),
        ("evaluated/spark_wrong.jsonl", "generated"),
        ("generated/spark_base.jsonl", "base"),
        ("generated/spark_empty.jsonl", ""),
        (".gitignore", "custom ignore rules\n"),
    ],
)
def test_unexpected_contents_are_preserved(
    tmp_path: Path,
    smoke_evaluator: MagicMock,
    capsys: pytest.CaptureFixture[str],
    relative_path: str,
    contents: str,
):
    assert run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    if contents == "generated":
        (dataset_file,) = (tmp_path / "generated").glob("*.jsonl")
        contents = dataset_file.read_text()
    elif contents == "base":
        sample = load_sample(EXPANDED_DATASETS_DIR, DEFAULT_SMOKE_SAMPLE).samples[0]
        contents = sample.model_dump_json(exclude_defaults=True) + "\n"
    unexpected = tmp_path / relative_path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text(contents)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
    )
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert "✗ Prepare results directory" in capsys.readouterr().out
    assert smoke_evaluator.call_count == 3


@pytest.mark.parametrize("symlink_kind", ["root", "dataset-directory", "dataset-file"])
def test_output_symlinks_are_preserved(
    tmp_path: Path, smoke_evaluator: MagicMock, symlink_kind: str
):
    output_dir = tmp_path / "results"
    assert run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=output_dir,
    )
    if symlink_kind == "root":
        link = tmp_path / "linked-results"
        link.symlink_to(output_dir, target_is_directory=True)
        output_dir = link
    elif symlink_kind == "dataset-directory":
        link = output_dir / "generated"
        target = tmp_path / "saved-generated"
        link.rename(target)
        link.symlink_to(target, target_is_directory=True)
    else:
        (link,) = (output_dir / "generated").glob("*.jsonl")
        target = tmp_path / "saved-dataset.jsonl"
        link.rename(target)
        link.symlink_to(target)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=output_dir,
    )
    assert link.is_symlink()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert smoke_evaluator.call_count == 3


def test_canonical_failure_preserves_output(
    tmp_path: Path, smoke_evaluator: MagicMock, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "old.txt").write_text("old results")
    (evaluated,) = smoke_evaluator.return_value
    sample = evaluated.samples[0].model_copy(
        update={
            "evaluation_results": [
                EvaluationStatsFailed(eval=Eval.BUILD, exception="broken evaluator")
            ]
        }
    )
    smoke_evaluator.return_value = [replace(evaluated, samples=[sample])]
    assert not run_smoke_test(
        tool_config=DUMMY_CONFIG,
        datasets_dir=EXPANDED_DATASETS_DIR,
        output_dir=tmp_path,
        allow_unsolved=True,
    )
    assert "✗ Evaluate canonical solution" in capsys.readouterr().out
    assert (tmp_path / "old.txt").read_text() == "old results"
    assert list(tmp_path.iterdir()) == [tmp_path / "old.txt"]


@pytest.mark.parametrize(
    ("result", "allow_unsolved", "expected_pass", "description"),
    [
        (
            EvaluationStatsBuild(
                compiled=False, pre_format_warnings=False, post_format_warnings=False
            ),
            True,
            True,
            "build failed",
        ),
        (
            EvaluationStatsBuild(
                compiled=False, pre_format_warnings=False, post_format_warnings=False
            ),
            False,
            False,
            "the sample was not solved: build failed",
        ),
        (
            EvaluationStatsFailed(eval=Eval.BUILD, exception="broken evaluator"),
            True,
            False,
            "build errored (broken evaluator)",
        ),
        (
            EvaluationStatsTimedOut(
                eval=Eval.BUILD, cmd_timed_out=["gprbuild"], timeout=1
            ),
            True,
            False,
            "build timed out after 1.0s",
        ),
    ],
    ids=["unsolved-allowed", "unsolved-rejected", "evaluator-error", "timeout"],
)
def test_generation_evaluation_verdict(
    tmp_path: Path,
    smoke_evaluator: MagicMock,
    capsys: pytest.CaptureFixture[str],
    result: EvaluationStats,
    *,
    allow_unsolved: bool,
    expected_pass: bool,
    description: str,
):
    (canonical,) = smoke_evaluator.return_value
    sample = canonical.samples[0].model_copy(update={"evaluation_results": [result]})
    smoke_evaluator.side_effect = [[canonical], [replace(canonical, samples=[sample])]]
    assert (
        run_smoke_test(
            tool_config=DUMMY_CONFIG,
            datasets_dir=EXPANDED_DATASETS_DIR,
            output_dir=tmp_path,
            allow_unsolved=allow_unsolved,
        )
        is expected_pass
    )
    out = capsys.readouterr().out
    mark = "✓" if expected_pass else "✗"
    verdict = "PASSED" if expected_pass else "FAILED"
    assert f"{mark} Evaluate generation" in out
    assert description in out
    assert f"Smoke test {verdict}" in out
    assert list((tmp_path / "evaluated").glob("*.jsonl"))


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
    assert (tmp_path / "evaluated").is_dir()
