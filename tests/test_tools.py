import shutil
import subprocess
import textwrap
from logging import ERROR, WARN
from pathlib import Path
from typing import ClassVar

import pytest
from helpers import (
    assert_log,
    compacted_test_datasets,  # noqa: F401  # Fixtures used implicitly
    expanded_test_datasets,  # noqa: F401  # Fixtures used implicitly
    generated_test_datasets,  # noqa: F401  # Fixtures used implicitly
)

from ada_eval.datasets.loader import load_datasets
from ada_eval.datasets.types.datasets import (
    Dataset,
    dataset_has_sample_type,
)
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.samples import (
    GENERATED_SAMPLE_TYPES,
    INITIAL_SAMPLE_TYPES,
    AdaSample,
    ExplainSample,
    GeneratedAdaSample,
    GeneratedExplainSample,
    GeneratedSample,
    GeneratedSparkSample,
    GenerationStats,
    Sample,
    SampleKind,
    SparkSample,
)
from ada_eval.tools import Tool, create_tool
from ada_eval.tools.generic_tool import BaseConfig, GenericTool
from ada_eval.tools.shell_script import ShellScriptConfig


def check_progress_bar(capsys: pytest.CaptureFixture[str], total: int, tool_name: str):
    output = capsys.readouterr()
    assert f"Generating completions with {tool_name}: 100%" in output.err
    assert f" {total}/{total} " in output.err
    assert output.out == ""


def test_generic_tool(
    tmp_path: Path,
    compacted_test_datasets: Path,  # noqa: F811  # pytest fixture
    expanded_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Create a mock tool
    mock_generation_stats = GenerationStats(
        exit_code=0,
        stdout="This is the stdout",
        stderr="This is the stderr",
        runtime_ms=123,
    )
    mock_ada_solution = DirectoryContents(
        {Path("generated_file"): "This is a generated file."}
    )
    mock_explain_solution = "This is a generated explanation."

    class MockTool0(GenericTool[BaseConfig, Sample, GeneratedSample]):
        name: ClassVar = "mock_tool_0"
        type_map: ClassVar = {
            INITIAL_SAMPLE_TYPES[k]: GENERATED_SAMPLE_TYPES[k] for k in SampleKind
        }
        config_type = BaseConfig

        def apply(self, sample: Sample) -> GeneratedSample:
            if isinstance(sample, ExplainSample):
                generated_solution: object = mock_explain_solution
            else:
                generated_solution = mock_ada_solution
            return GENERATED_SAMPLE_TYPES[sample.kind](
                **sample.model_dump(),
                generation_stats=mock_generation_stats,
                generated_solution=generated_solution,
            )

    # Instantiate the mock tool from a config file
    tmp_config_file = tmp_path / "mock_tool_config.json"
    tmp_config_file.write_text('{"timeout_s": 10}')
    tool = MockTool0.from_config_file(tmp_config_file)

    # Test applying the tool to the test datasets
    base_datasets = load_datasets(compacted_test_datasets)
    generated_datasets_0, failed_datasets_0, incompatible_datasets_0 = (
        tool.apply_to_datasets(base_datasets, jobs=8)
    )
    check_progress_bar(capsys, 5, "mock_tool_0")

    def check_generated_datasets(generated_datasets: list[Dataset[GeneratedSample]]):
        for dataset in generated_datasets:
            base_dataset = next(
                d for d in base_datasets if d.dirname == dataset.dirname
            )
            assert dataset_has_sample_type(dataset, GeneratedSample)
            for sample in dataset.samples:
                base_sample = next(
                    s for s in base_dataset.samples if s.name == sample.name
                )
                assert sample.generation_stats == mock_generation_stats
                if dataset.kind == SampleKind.EXPLAIN:
                    assert isinstance(sample, GeneratedExplainSample)
                    assert sample.generated_solution == mock_explain_solution
                else:
                    assert isinstance(sample, GeneratedAdaSample)
                    assert sample.generated_solution == mock_ada_solution
                assert base_sample.model_dump() == sample.model_dump(
                    exclude={"generation_stats", "generated_solution"}
                )

    # Check the mock generations have been added successfully
    assert len(failed_datasets_0) == 0
    assert len(incompatible_datasets_0) == 0
    assert len(generated_datasets_0) == 3
    check_generated_datasets(generated_datasets_0)

    # Test applying directly to a directory (both packed and unpacked)
    out_dir = tmp_path / "output"
    for in_dir in (compacted_test_datasets, expanded_test_datasets):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        assert not out_dir.exists()
        tool.apply_to_directory(in_dir, out_dir, jobs=8)
        check_progress_bar(capsys, 5, "mock_tool_0")
        generated_datasets = [  # Mypy-friendly check that all are `GeneratedSample`s
            d
            for d in load_datasets(out_dir)
            if dataset_has_sample_type(d, GeneratedSample)
        ]
        assert len(generated_datasets) == 3
        check_generated_datasets(generated_datasets)

    # None of the above should have logged any warnings
    assert caplog.text == ""

    # Define a new mock tool to be incompatible with Explain datasets and to
    # raise an exception on `test_sample_0` of the Spark dataset
    class MockTool1(GenericTool[BaseConfig, AdaSample, GeneratedAdaSample]):
        name: ClassVar = "mock_tool_1"
        type_map: ClassVar = {
            AdaSample: GeneratedAdaSample,
            SparkSample: GeneratedSparkSample,
        }
        config_type = BaseConfig

        def apply(self, sample: AdaSample) -> GeneratedAdaSample:
            if isinstance(sample, SparkSample) and sample.name == "test_sample_0":
                raise RuntimeError("Mock failure on test_sample_0")
            gen_sample = GENERATED_SAMPLE_TYPES[sample.kind](
                **sample.model_dump(),
                generation_stats=mock_generation_stats,
                generated_solution=mock_ada_solution,
            )
            assert isinstance(gen_sample, GeneratedAdaSample)
            return gen_sample

    # Test applying the new mock tool to the test datasets
    tool1 = MockTool1.from_config_file(tmp_config_file)
    generated_datasets_1, failed_datasets_1, incompatible_datasets_1 = (
        tool1.apply_to_datasets(base_datasets, jobs=8)
    )
    check_progress_bar(capsys, 4, "mock_tool_1")  # 4 because Explain sample is excluded

    # The failure should have been logged with a full stack trace
    def check_fail_log() -> None:
        msg = "Error processing sample test_sample_0 from dataset spark_test"
        fail_log = assert_log(caplog, ERROR, msg)
        assert fail_log.exc_text.endswith("RuntimeError: Mock failure on test_sample_0")

    check_fail_log()
    caplog.clear()

    # Check that the explain dataset is recognised as incompatible
    assert len(incompatible_datasets_1) == 1
    assert incompatible_datasets_1[0].dirname == "explain_test"
    assert incompatible_datasets_1[0].sample_type is ExplainSample

    # Check that the exception on the spark sample is recorded properly in
    # `failed_datasets_1`
    assert len(failed_datasets_1) == 1
    assert failed_datasets_1[0].dirname == "spark_test"
    assert len(failed_datasets_1[0].samples) == 1
    failed_sample = failed_datasets_1[0].samples[0]
    assert isinstance(failed_sample, SparkSample)
    assert not isinstance(failed_sample, GeneratedSparkSample)
    base_spark_sample_0 = next(
        s
        for d in base_datasets
        if d.dirname == "spark_test"
        for s in d.samples
        if s.name == "test_sample_0"
    )
    assert failed_sample == base_spark_sample_0

    # Check that the other samples have been generated correctly
    assert len(generated_datasets_1) == 2
    assert sum(len(d.samples) for d in generated_datasets_1) == 3
    check_generated_datasets(list(generated_datasets_1))

    # Test applying the new tool directly to a directory (existing files should
    # be overwritten, so `explain_test.jsonl` should be removed)
    assert (out_dir / "explain_test.jsonl").is_file()
    tool1.apply_to_directory(compacted_test_datasets, out_dir, jobs=8)
    assert not (out_dir / "explain_test.jsonl").exists()
    check_progress_bar(capsys, 4, "mock_tool_1")

    # The failure should have been logged again, and there should also be
    # warnings about the omissions from the output
    check_fail_log()
    msg = (
        "'mock_tool_1' is incompatible with 1 datasets found at "
        f"'{compacted_test_datasets}'. These datasets will be omitted from the results."
    )
    assert_log(caplog, WARN, msg)
    msg = (
        "'mock_tool_1' failed on 1 samples found at "
        f"'{compacted_test_datasets}'. These samples will be omitted from the results."
    )
    assert_log(caplog, WARN, msg)
    caplog.clear()

    # Check that the generated datasets are as expected
    generated_datasets = [
        d for d in load_datasets(out_dir) if dataset_has_sample_type(d, GeneratedSample)
    ]
    assert len(generated_datasets) == 2
    assert sum(len(d.samples) for d in generated_datasets) == 3
    check_generated_datasets(generated_datasets)

    # Check that there is a warning when nothing is generated
    (compacted_test_datasets / "ada_test.jsonl").unlink()
    (compacted_test_datasets / "spark_test.jsonl").unlink()
    tool1.apply_to_directory(compacted_test_datasets, out_dir, jobs=8)
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == ""
    assert_log(caplog, WARN, "No datasets compatible with mock_tool_1 found.")
    msg = (
        "'mock_tool_1' could not be applied to any samples found at "
        f"'{compacted_test_datasets}'."
    )
    assert_log(caplog, WARN, msg)


@pytest.mark.skipif(not shutil.which("sh"), reason="sh not available")
def test_shell_script(
    tmp_path: Path,
    compacted_test_datasets: Path,  # noqa: F811  # pytest fixture
    generated_test_datasets: Path,  # noqa: F811  # pytest fixture
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    # Create a mock shell script (which just `echo`s and writes to
    # `./generated_file`) and tool config file
    script_dir = tmp_path / "mock_script"
    script_dir.mkdir()
    shell_script = script_dir / "mock_script.sh"
    shell_script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env sh
            echo "This file was added during generation" > generated_file
            echo "This is the generation's stdout"
            """
        )
    )
    shell_script.chmod(0o700)  # Make it executable
    config_dir = tmp_path / "mock_tool_config"
    config_dir.mkdir()
    config_file = config_dir / "mock_tool_config.json"
    shell_script_relative = shell_script.relative_to(config_dir, walk_up=True)
    config_file.write_text(
        f'{{"timeout_s": 1, "shell_script": "{shell_script_relative}"}}'
    )

    # Verify the script works. This also serves to trigger any first-run
    # Gatekeeper checks or similar so that the actual tool calls do not time out.
    script_test_dir = tmp_path / "script_test"
    script_test_dir.mkdir()
    result = subprocess.run(
        [str(shell_script), "dummy_arg"],
        cwd=script_test_dir,
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    assert result.stdout == "This is the generation's stdout\n"
    assert result.stderr == ""
    generated_file = script_test_dir / "generated_file"
    assert generated_file.read_text() == "This file was added during generation\n"

    # Run the tool on the test datasets
    base_datasets = load_datasets(compacted_test_datasets)
    tool = create_tool(Tool.SHELL_SCRIPT, config_file)
    assert isinstance(tool.config, ShellScriptConfig)
    generated_datasets, failed_datasets, incompatible_datasets = tool.apply_to_datasets(
        base_datasets, jobs=8
    )

    # Only the Spark dataset should be compatible
    check_progress_bar(capsys, 3, "shell_script")
    assert len(failed_datasets) == 0
    assert len(incompatible_datasets) == 2
    assert set(incompatible_datasets) == {  # Hash/Equality ignores `samples`
        Dataset(name="test", sample_type=ExplainSample, samples=[]),
        Dataset(name="test", sample_type=AdaSample, samples=[]),
    }

    # The generated `spark_test` dataset should match that in the
    # `generated_test_datasets` fixture, with the sole exception of the
    # `generation_stats`'s `runtime_ms`, which may be non-zero.
    generated_datasets_fixture = load_datasets(generated_test_datasets)
    generated_spark_dataset_fixture = next(
        d for d in generated_datasets_fixture if d.kind == SampleKind.SPARK
    )
    assert len(generated_datasets) == 1
    generated_spark_dataset = generated_datasets[0]
    assert generated_spark_dataset.sample_type is GeneratedSparkSample
    assert generated_spark_dataset.name == "test"
    for sample in generated_spark_dataset.samples:
        assert sample.generation_stats.runtime_ms in range(100)
        sample.generation_stats.runtime_ms = 0  # Normalize for comparison
    assert generated_spark_dataset.samples == generated_spark_dataset_fixture.samples

    # Test timeout handling
    shell_script.write_text("#!/usr/bin/env sh\nsleep 20\n")
    generated_datasets, failed_datasets, incompatible_datasets = tool.apply_to_datasets(
        base_datasets, jobs=8
    )
    output = capsys.readouterr()
    assert "Generating completions with shell_script:" in output.err
    assert output.out == ""
    assert len(failed_datasets) == 0
    assert len(incompatible_datasets) == 2
    assert len(generated_datasets) == 1
    generated_spark_dataset = generated_datasets[0]
    assert generated_spark_dataset.sample_type is GeneratedSparkSample
    assert generated_spark_dataset.name == "test"
    for sample in generated_spark_dataset.samples:
        assert sample.generation_stats.exit_code == 124
        assert sample.generation_stats.stdout == ""
        assert sample.generation_stats.stderr == "Process timed out after 1 seconds"
        assert sample.generation_stats.runtime_ms in range(950, 1050)
        assert sample.generated_solution == sample.sources

    # Nothing should have been logged, since nothing failed
    assert caplog.text == ""
