import json
import re
from pathlib import Path

import pytest

from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.exit_status import ExitStatus
from ada_eval.datasets.types.samples import (
    EVALUATED_SAMPLE_TYPES,
    GENERATED_SAMPLE_TYPES,
    INITIAL_SAMPLE_TYPES,
    GeneratedProofSample,
    GenerationStats,
    Location,
    PathMustBeRelativeError,
    ProofSample,
    SampleKind,
    SampleStage,
    SparkSample,
    SubprogramNotFoundError,
)

TEST_ADB = """\
package body Some_Name is

   Some_Name : constant String := "Some_Value";

   function Some_Name_With_Suffix return string is
   begin
      return "Return Value";
   end Some_Name_With_Suffix;

   function Some_Name (I : Integer) return Integer is
   begin
      return I + 1;
   end Some_Name;

   procedure Some_Name is null;

end Some_Name;
"""


def test_location_absolute_path():
    error_msg = "Path '/absolute/path' is not relative"
    with pytest.raises(PathMustBeRelativeError, match=re.escape(error_msg)):
        Location(path=Path("/absolute/path"), subprogram_name="Some_Name")


def test_location_find_line_number(tmp_path: Path):
    # Write the test `.adb` file
    test_adb = tmp_path / "some_name.adb"
    test_adb.write_text(TEST_ADB)
    # Check that `find_line_number()` finds the first instance of `Some_Name`
    # as a subprogram (i.e. line 10)
    loc = Location(path=test_adb.relative_to(tmp_path), subprogram_name="Some_Name")
    assert loc.find_line_number(tmp_path) == 10
    # Check that it works from a different path
    (tmp_path / "src").mkdir()
    test_adb = test_adb.rename(tmp_path / "src" / "some_name.adb")
    loc.path = test_adb.relative_to(tmp_path)
    assert loc.find_line_number(tmp_path) == 10
    # Check that procedures are also recognised as subprograms
    new_content = re.sub(
        r"function Some_Name .*?end Some_Name;", "", TEST_ADB, flags=re.DOTALL
    )
    test_adb.write_text(new_content)
    assert loc.find_line_number(tmp_path) == 12  # 2 lines of residual whitespace
    # Check that it raises an error if no subprogram is found
    new_content = new_content.replace("procedure Some_Name is null;", "")
    test_adb.write_text(new_content)
    error_msg = f"Subprogram 'Some_Name' not found in '{test_adb}'"
    with pytest.raises(SubprogramNotFoundError, match=re.escape(error_msg)):
        loc.find_line_number(tmp_path)


def test_sample_kind_str():
    assert str(SampleKind.ADA) == "ada"
    assert str(SampleKind.EXPLAIN) == "explain"
    assert str(SampleKind.PROOF) == "proof"
    assert str(SampleKind.SPARK) == "spark"


def test_sample_type_dicts_are_complete():
    for kind in SampleKind:
        assert kind in INITIAL_SAMPLE_TYPES
        assert INITIAL_SAMPLE_TYPES[kind].kind == kind
        assert INITIAL_SAMPLE_TYPES[kind].stage == SampleStage.INITIAL
        assert kind in GENERATED_SAMPLE_TYPES
        assert GENERATED_SAMPLE_TYPES[kind].kind == kind
        assert GENERATED_SAMPLE_TYPES[kind].stage == SampleStage.GENERATED
        assert kind in EVALUATED_SAMPLE_TYPES
        assert EVALUATED_SAMPLE_TYPES[kind].kind == kind
        assert EVALUATED_SAMPLE_TYPES[kind].stage == SampleStage.EVALUATED


class TestGenerationStatsBackwardCompat:
    """Test backward compatibility for GenerationStats."""

    def test_exit_code_converts_to_exit_status(self):
        """Test that exit_code int is converted to ExitStatus."""
        # Simulate loading old data with exit_code
        stats = GenerationStats(
            exit_status=0,  # Will be converted by validator
            stdout="output",
            stderr="",
            runtime_ms=1000,
        )
        assert stats.exit_status == ExitStatus.SUCCESS

    def test_exit_code_124_converts_to_timeout(self):
        """Test that exit_code 124 becomes TIMEOUT."""
        stats = GenerationStats(
            exit_status=124,
            stdout="",
            stderr="timeout",
            runtime_ms=5000,
        )
        assert stats.exit_status == ExitStatus.TIMEOUT

    def test_exit_code_nonzero_converts_to_process_error(self):
        """Test that non-zero exit_code becomes PROCESS_ERROR."""
        stats = GenerationStats(
            exit_status=1,
            stdout="",
            stderr="error",
            runtime_ms=100,
        )
        assert stats.exit_status == ExitStatus.PROCESS_ERROR

    def test_exit_status_string_accepted(self):
        """Test that ExitStatus string values work."""
        stats = GenerationStats(
            exit_status="success",
            stdout="",
            stderr="",
            runtime_ms=100,
        )
        assert stats.exit_status == ExitStatus.SUCCESS


def test_spark_sample_expected_non_spark_entities_round_trip(tmp_path: Path):
    sample = SparkSample(
        name="stubbed",
        location=Location(path=Path("src/p.ads"), subprogram_name="P"),
        prompt="",
        comments="",
        sources=DirectoryContents({}),
        canonical_solution=DirectoryContents({}),
        canonical_evaluation_results=[],
        unit_tests=DirectoryContents({}),
        expected_non_spark_entities=["P.Helper"],
    )
    sample.unpack(tmp_path)
    other = json.loads((tmp_path / "stubbed" / "other.json").read_text())
    assert other["expected_non_spark_entities"] == ["P.Helper"]
    assert SparkSample.load_unpacked_sample(tmp_path / "stubbed") == sample


def test_proof_sample_round_trip_has_no_tests_dir(tmp_path: Path):
    sample = ProofSample(
        name="proof_only",
        location=Location(path=Path("src/p.ads"), subprogram_name="P"),
        prompt="",
        comments="",
        sources=DirectoryContents({Path("main.gpr"): b"project Main is end Main;"}),
        canonical_solution=DirectoryContents(
            {Path("main.gpr"): b"project Main is end Main;"}
        ),
        canonical_evaluation_results=[],
        expected_non_spark_entities=["P.Helper"],
    )
    sample.unpack(tmp_path)
    assert not (tmp_path / "proof_only" / "tests").exists()
    other = json.loads((tmp_path / "proof_only" / "other.json").read_text())
    assert "prove_whole_project" not in other
    assert ProofSample.load_unpacked_sample(tmp_path / "proof_only") == sample
    whole = sample.model_copy(update={"name": "whole", "prove_whole_project": True})
    whole.unpack(tmp_path)
    other = json.loads((tmp_path / "whole" / "other.json").read_text())
    assert other["prove_whole_project"] is True
    assert ProofSample.load_unpacked_sample(tmp_path / "whole") == whole


def test_solution_to_evaluate_uses_the_canonical_project_files():
    sample = GeneratedProofSample(
        name="pinned",
        location=Location(path=Path("src/p.ads"), subprogram_name="P"),
        prompt="",
        comments="",
        sources=DirectoryContents({}),
        canonical_solution=DirectoryContents(
            {
                Path("main.gpr"): b"canonical",
                Path("gnat.adc"): b"pragma SPARK_Mode;",
                Path("src/p.adb"): b"canonical body",
            }
        ),
        canonical_evaluation_results=[],
        generation_stats=GenerationStats(
            exit_status=ExitStatus.SUCCESS, stdout="", stderr="", runtime_ms=0
        ),
        generated_solution=DirectoryContents(
            {
                Path("main.gpr"): b"generated",
                Path("other.gpr"): b"generated",
                Path("src/p.adb"): b"generated body",
            }
        ),
    )
    assert sample.solution_to_evaluate().files == {
        Path("main.gpr"): b"canonical",
        Path("gnat.adc"): b"pragma SPARK_Mode;",
        Path("src/p.adb"): b"generated body",
    }
