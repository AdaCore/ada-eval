import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ada_eval.datasets.loader import load_datasets, load_packed_dataset
from ada_eval.datasets.trivial_generations import generate_canonical
from ada_eval.datasets.types import (
    Dataset,
    Eval,
    EvaluatedSample,
    EvaluationStatsFailed,
    EvaluationStatsTimedOut,
    ExitStatus,
    GeneratedSample,
    Sample,
    SampleStage,
    save_datasets,
)
from ada_eval.datasets.types.execution_log import ReasoningLogEntry
from ada_eval.evals import evaluate_datasets
from ada_eval.paths import SMOKE_TEST_DIR, TOOL_CONFIGS_DIR
from ada_eval.tools import create_tool_from_config
from ada_eval.tools.generic_tool import GenericTool

DEFAULT_SMOKE_SAMPLE = "spark_learn/show_uninitialized"

REQUIRED_EXECUTABLES = ("gprbuild", "gprclean", "gprls", "gnatformat", "gnatprove")


class SmokeTestFailedError(Exception):
    """A required part of the smoke test failed."""


def load_sample(datasets_dir: Path, sample_id: str) -> Dataset[Sample]:
    """Load a specific sample from the given datasets directory."""
    dataset_dirname, _, sample_name = sample_id.partition("/")
    if not sample_name or not (datasets_dir / dataset_dirname / sample_name).is_dir():
        msg = f"no sample '{sample_id}' in '{datasets_dir}'"
        raise SmokeTestFailedError(msg)
    (dataset,) = load_datasets(datasets_dir / dataset_dirname)
    samples = [s for s in dataset.samples if s.name == sample_name]
    return Dataset(name=dataset.name, sample_type=dataset.sample_type, samples=samples)


def describe_results(sample: EvaluatedSample) -> tuple[bool, bool, str]:
    """Summarize the results of the evaluations for a sample."""
    parts = []
    for result in sample.evaluation_results:
        if isinstance(result, EvaluationStatsFailed):
            parts.append(f"{result.eval} errored ({result.exception})")
        elif isinstance(result, EvaluationStatsTimedOut):
            parts.append(f"{result.eval} timed out after {result.timeout}s")
        else:
            parts.append(f"{result.eval} {'passed' if result.passed else 'failed'}")
    all_ran = not any(
        isinstance(r, EvaluationStatsFailed | EvaluationStatsTimedOut)
        for r in sample.evaluation_results
    )
    all_passed = all(r.passed for r in sample.evaluation_results)
    return all_ran, all_passed, ", ".join(parts)


def last_output_line(sample: GeneratedSample, max_length: int = 300) -> str | None:
    """Prefer the last reasoning line, then stderr, then stdout."""
    stats = sample.generation_stats
    reasoning = [
        e.text for e in stats.execution_log if isinstance(e, ReasoningLogEntry)
    ]
    for output in (*reasoning[-1:], stats.stderr, stats.stdout):
        lines = output.strip().splitlines()
        if lines:
            line = lines[-1]
            return line if len(line) <= max_length else line[:max_length] + "..."
    return None


def default_output_dir(tool_config: Path) -> Path:
    """Determine the default output directory for a given tool configuration."""
    try:
        relative = tool_config.resolve().relative_to(TOOL_CONFIGS_DIR)
    except ValueError:
        relative = Path(tool_config.name)
    return SMOKE_TEST_DIR / relative.with_suffix("")


def validate_output_dir(output_dir: Path) -> None:
    """Check every entry before allowing previous results to be removed."""
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        msg = f"refusing to clear '{output_dir}': not a plain directory"
        raise SmokeTestFailedError(msg)
    if not output_dir.exists():
        return
    stages = {"generated": SampleStage.GENERATED, "evaluated": SampleStage.EVALUATED}
    for entry in output_dir.iterdir():
        if (
            not entry.is_symlink()
            and entry.name == ".gitignore"
            and entry.is_file()
            and entry.read_text() == "*\n"
        ):
            continue
        if entry.is_symlink() or entry.name not in stages or not entry.is_dir():
            msg = f"refusing to clear '{output_dir}': unexpected entry '{entry}'"
            raise SmokeTestFailedError(msg)
        for dataset_file in entry.iterdir():
            if (
                dataset_file.is_symlink()
                or not dataset_file.is_file()
                or dataset_file.suffix != ".jsonl"
            ):
                msg = (
                    f"refusing to clear '{output_dir}': "
                    f"unexpected entry '{dataset_file}'"
                )
                raise SmokeTestFailedError(msg)
            dataset = load_packed_dataset(dataset_file)
            if not dataset.samples or dataset.stage != stages[entry.name]:
                msg = (
                    f"refusing to clear '{output_dir}': '{dataset_file}' "
                    f"is not a nonempty {entry.name} dataset"
                )
                raise SmokeTestFailedError(msg)


class SmokeTest:
    tool: GenericTool[Any, Any, Any]
    dataset: Dataset[Sample]
    generated: Dataset[GeneratedSample]

    def __init__(
        self,
        tool_config: Path,
        datasets_dir: Path,
        output_dir: Path,
        sample_id: str,
        *,
        allow_unsolved: bool,
    ):
        self.tool_config = tool_config
        self.datasets_dir = datasets_dir
        self.output_dir = output_dir
        self.output_ready = False
        self.sample_id = sample_id
        self.allow_unsolved = allow_unsolved

    def load_config(self) -> str:
        self.tool = create_tool_from_config(self.tool_config)
        return f"{self.tool.name} from '{self.tool_config}'"

    def find_executables(self) -> str:
        missing = [e for e in REQUIRED_EXECUTABLES if shutil.which(e) is None]
        if missing:
            msg = f"not on PATH: {', '.join(missing)}"
            raise SmokeTestFailedError(msg)
        return ", ".join(REQUIRED_EXECUTABLES)

    def load_sample(self) -> str:
        self.dataset = load_sample(self.datasets_dir, self.sample_id)
        return self.sample_id

    def evaluate_canonical(self) -> str:
        """Check the toolchain against a known solution."""
        (evaluated,) = evaluate_datasets(
            list(Eval), generate_canonical([self.dataset]), jobs=1
        )
        all_ran, all_passed, description = describe_results(evaluated.samples[0])
        if not (all_ran and all_passed):
            msg = f"the canonical solution should pass every eval, but {description}"
            raise SmokeTestFailedError(msg)
        return description

    def prepare_output_dir(self) -> str:
        validate_output_dir(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("generated", "evaluated"):
            directory = self.output_dir / name
            if directory.exists():
                shutil.rmtree(directory)
        (self.output_dir / ".gitignore").write_text("*\n")
        self.output_ready = True
        return str(self.output_dir)

    def generate(self) -> str:
        (sample,) = self.dataset.samples
        timeout_s = self.tool.config.timeout_s
        print(f"  Running {self.tool.name} (timeout {timeout_s}s)...", flush=True)
        generated_sample = self.tool.apply(sample)
        self.generated = Dataset(
            name=self.dataset.name,
            sample_type=type(generated_sample),
            samples=[generated_sample],
        )
        save_datasets([self.generated], self.output_dir / "generated")
        stats = generated_sample.generation_stats
        description = f"exit status {stats.exit_status}"
        if stats.total_cost_usd is not None:
            description += f", cost ${stats.total_cost_usd:.4f}"
        if stats.exit_status != ExitStatus.SUCCESS:
            last_output = last_output_line(generated_sample)
            if last_output:
                description += f", last output: {last_output}"
            raise SmokeTestFailedError(description)
        return description

    def evaluate_generation(self) -> str:
        (evaluated,) = evaluate_datasets(list(Eval), [self.generated], jobs=1)
        save_datasets([evaluated], self.output_dir / "evaluated")
        all_ran, all_passed, description = describe_results(evaluated.samples[0])
        if not all_ran:
            raise SmokeTestFailedError(description)
        if not all_passed and not self.allow_unsolved:
            msg = f"the sample was not solved: {description}"
            raise SmokeTestFailedError(msg)
        return description

    def run(self) -> bool:
        stages: list[tuple[str, Callable[[], str]]] = [
            ("Load tool config", self.load_config),
            ("Find Ada/SPARK tools", self.find_executables),
            ("Load sample", self.load_sample),
            ("Evaluate canonical solution", self.evaluate_canonical),
            ("Prepare results directory", self.prepare_output_dir),
            ("Generate", self.generate),
            ("Evaluate generation", self.evaluate_generation),
        ]
        start = time.monotonic()
        passed = True
        for name, stage in stages:
            stage_start = time.monotonic()
            try:
                description = stage()
            except SmokeTestFailedError as e:
                passed, description = False, str(e)
            except Exception as e:  # noqa: BLE001 (any failure is reported)
                passed, description = False, f"{type(e).__name__}: {e}"
            mark = "✓" if passed else "✗"
            duration_s = time.monotonic() - stage_start
            print(f"{mark} {name} ({duration_s:.1f}s): {description}", flush=True)
            if not passed:
                break
        verdict = "PASSED" if passed else "FAILED"
        summary = f"\nSmoke test {verdict} in {time.monotonic() - start:.0f}s"
        if self.output_ready:
            summary += f" (results in '{self.output_dir}')"
        print(summary)
        return passed


def run_smoke_test(
    tool_config: Path,
    datasets_dir: Path,
    output_dir: Path,
    sample_id: str = DEFAULT_SMOKE_SAMPLE,
    *,
    allow_unsolved: bool = False,
) -> bool:
    return SmokeTest(
        tool_config, datasets_dir, output_dir, sample_id, allow_unsolved=allow_unsolved
    ).run()
