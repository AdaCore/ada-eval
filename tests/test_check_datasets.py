import re
import shutil
from logging import ERROR, INFO
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from helpers import assert_log

from ada_eval.check_datasets import (
    EditableFileError,
    ExistingAssertionError,
    FailedCanonicalEvaluationError,
    InaccurateCanonicalEvaluationError,
    InvalidBaselineEvaluationError,
    NonTerminatingSolutionPassedError,
    ObjectDirError,
    PassedBaselineEvaluationError,
    ProofRuleSrcPatternError,
    TrustedEntityInEditableFileError,
    UnexpectedNonSparkEntitiesError,
    check_base_datasets,
    check_editable_files,
    check_editable_procedures_terminate,
    check_expected_non_spark_entities,
    check_object_dirs,
    check_src_patterns,
    with_infinite_loop,
)
from ada_eval.datasets import (
    Dataset,
    Eval,
    EvaluationStatsBuild,
    EvaluationStatsProve,
    ExitStatus,
    ProofCheck,
    Sample,
    dataset_has_sample_type,
    load_datasets,
    save_datasets,
)
from ada_eval.datasets.trivial_generations import generate_canonical
from ada_eval.datasets.types.datasets import DatasetsMismatchError
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.datasets.types.samples import (
    EvaluatedSparkSample,
    ExplainSample,
    GeneratedProofSample,
    GeneratedSparkSample,
    GenerationStats,
    Location,
    ProofSample,
    SparkSample,
)
from ada_eval.evals.evaluate import create_eval
from ada_eval.evals.generic_eval import GenericEval
from ada_eval.evals.prove import Prove


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprclean"), reason="gprclean not available")
@pytest.mark.skipif(not shutil.which("gnatformat"), reason="gnatformat not available")
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_check_base_datasets(
    tmp_path: Path,
    check_test_datasets: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    caplog.set_level("INFO")

    # Load datasets consisting of one `SparkSample` which passes all checks, one
    # bad `SparkSample`, and one `ExplainSample` (which has some canonical
    # results it shouldn't, but is otherwise correct).
    datasets = load_datasets(check_test_datasets)
    explain_dataset = next(
        d for d in datasets if dataset_has_sample_type(d, ExplainSample)
    )
    explain_sample = explain_dataset.samples[0]
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    good_sample = next(s for s in spark_dataset.samples if s.name == "good")
    bad_sample = next(s for s in spark_dataset.samples if s.name == "bad")

    # Save this dataset in both expanded and compacted forms
    expanded_dir = tmp_path / "expanded"
    compacted_dir = tmp_path / "compacted"

    def save_both(
        expanded: list[Dataset[Sample]], compacted: list[Dataset[Sample]] | None = None
    ) -> None:
        if compacted is None:
            compacted = expanded
        save_datasets(expanded, expanded_dir, unpacked=True)
        save_datasets(compacted, compacted_dir, unpacked=False)

    save_both(datasets)

    # Check that a missing dataset is detected
    def run_check() -> None:
        check_base_datasets([expanded_dir, compacted_dir], jobs=8)

    shutil.copytree(expanded_dir / "spark_check", expanded_dir / "spark_other")
    error_msg = f"dataset 'spark_other' is only present in '{expanded_dir}'."
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that differing sample types are detected
    generated_sample = GeneratedSparkSample(
        **good_sample.model_dump(),
        generation_stats=GenerationStats(
            exit_status=ExitStatus.SUCCESS, stdout="", stderr="", runtime_ms=0
        ),
        generated_solution=DirectoryContents({}),
    )
    generated_dataset = Dataset(
        name="check", sample_type=GeneratedSparkSample, samples=[generated_sample]
    )
    save_both([spark_dataset, explain_dataset], [explain_dataset, generated_dataset])
    error_msg = (
        f"dataset 'spark_check' has type 'SparkSample' in '{expanded_dir}' but "
        f"type 'GeneratedSparkSample' in '{compacted_dir}'."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that missing samples are detected
    one_sample_dataset = Dataset(
        name="check",
        sample_type=SparkSample,
        samples=[good_sample],
    )
    save_both([explain_dataset, one_sample_dataset], [spark_dataset, explain_dataset])
    error_msg = (
        f"sample 'bad' in dataset 'spark_check' is only present in '{compacted_dir}'."
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that differing samples are detected
    modified_sample_0 = good_sample.model_copy(deep=True)
    modified_sample_0.canonical_solution.files[Path("src/foo.adb")] = b"expanded nested"
    modified_sample_0.sources.files[Path("new_file")] = b"new"
    modified_sample_0.prompt = "Modified prompt"
    modified_dataset_0 = Dataset(
        name="check", sample_type=SparkSample, samples=[bad_sample, modified_sample_0]
    )
    modified_sample_1 = good_sample.model_copy(deep=True)
    modified_sample_1.canonical_solution.files[Path("src/foo.adb")] = (
        b"compacted nested"
    )
    modified_dataset_1 = Dataset(
        name="check", sample_type=SparkSample, samples=[modified_sample_1, bad_sample]
    )
    save_both([modified_dataset_0], [modified_dataset_1])
    error_msg = (
        "sample 'good' in dataset 'spark_check' differs between "
        f"'{expanded_dir}' and '{compacted_dir}':\n\n"
        "{'prompt': 'Modified prompt', 'sources': {PosixPath('new_file'): b'new'},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): b'expanded nested'}}"
        "\n\n{'prompt': '', 'sources': {},"
        " 'canonical_solution': {PosixPath('src/foo.adb'): b'compacted nested'}}"
    )
    with pytest.raises(DatasetsMismatchError, match=re.escape(error_msg)):
        run_check()

    # Check that non-passing canonical evaluation results are detected
    save_both(datasets)
    error_msg = (
        "sample 'bad' in dataset 'spark_check' has non-passing canonical "
        "evaluation results: ['prove', 'build', 'test']"
    )
    with pytest.raises(FailedCanonicalEvaluationError, match=re.escape(error_msg)):
        run_check()

    # Fix one of the canonical results and check that the message changes
    # appropriately
    bad_sample_build_stats = bad_sample.canonical_evaluation_results[1]
    assert isinstance(bad_sample_build_stats, EvaluationStatsBuild)
    bad_sample_build_stats.pre_format_warnings = False
    save_both(datasets)
    error_msg = (
        "sample 'bad' in dataset 'spark_check' has non-passing canonical "
        "evaluation results: ['prove', 'test']"
    )
    with pytest.raises(FailedCanonicalEvaluationError, match=re.escape(error_msg)):
        run_check()

    # Nothing should have been output or logged up to this point
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert caplog.text == ""

    # Fix the remaining canonical results and check that the discrepancy is
    # detected by re-evaluating
    bad_sample.canonical_evaluation_results = good_sample.canonical_evaluation_results
    save_both(datasets)
    error_msg = re.escape(
        "mismatch found on re-evaluating sample 'bad' in dataset 'spark_check':\n\n"
        "[{'pre_format_warnings': False}, {'result': 'proved', 'proved_checks': "
        "{'VC_POSTCONDITION': 1}, 'unproved_checks': {}}, {'passed_tests': True}]\n\n"
        "[{'pre_format_warnings': True}, {'result': 'unproved', 'proved_checks': {}, "
        "'unproved_checks': {'VC_POSTCONDITION': 1}}, {'passed_tests': False}]"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()

    # There should have been an info log and a loading bar for each evaluation
    reeval_msg = "Re-evaluating to check canonical evaluation results are accurate ..."
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()
    output = capsys.readouterr()
    for e in ("build", "prove", "test"):
        assert f"Evaluating with {e}" in output.err
    assert output.out == ""

    # Check that a missing canonical result yields a more readable error message
    bad_sample.canonical_evaluation_results = [
        good_sample.canonical_evaluation_results[1]
    ]
    save_both(datasets)
    error_msg = re.escape(
        "sample 'bad' in dataset 'spark_check' does not have the expected set of "
        "canonical evaluation results:\n['build'] != ['build', 'prove', 'test']"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()

    # Fix the actual issues with the bad sample's canonical solution and check
    # that the presence of a test result for the explain sample is detected.
    bad_sample.canonical_evaluation_results = good_sample.canonical_evaluation_results
    bad_sample.canonical_solution = good_sample.canonical_solution
    save_both(datasets)
    error_msg = re.escape(
        "sample 'explain' in dataset 'explain_check' does not have the "
        "expected set of canonical evaluation results:\n['test'] != []"
    )
    with pytest.raises(InaccurateCanonicalEvaluationError, match=error_msg):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert len(caplog.records) == 1
    caplog.clear()

    # Clear the `ExplainSample`'s canonical results and check that the passing
    # baseline evaluation of the 'bad' sample is detected
    explain_sample.canonical_evaluation_results = []
    save_both(datasets)
    error_msg = re.escape(
        "all evaluations passed on the unmodified sources of sample 'bad' in "
        "dataset 'spark_check'."
    )
    with pytest.raises(PassedBaselineEvaluationError, match=error_msg):
        run_check()
    baseline_msg = "Checking that null generations fail at least one eval ..."
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert len(caplog.records) == 2
    caplog.clear()

    # Modify the bad sample's `sources` to fail the evaluations and check that
    # `check_base_datasets()` raises no exceptions
    bad_sample.sources = good_sample.sources
    save_both(datasets)
    run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert_log(caplog, INFO, "Base datasets are correct.")
    assert len(caplog.records) == 3
    caplog.clear()

    # Mock `create_eval()` so that an exception is raised during evaluation of
    # the base sources (but not the canonical solutions), and check that this is
    # detected
    class MockEval(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
        """Mock build eval which raises if `src/foo.adb` contains 'return 1'."""

        eval: ClassVar = Eval.BUILD
        supported_types: ClassVar = {GeneratedSparkSample: EvaluatedSparkSample}

        def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsBuild:
            if b"return 1" in sample.generated_solution.files[Path("src/foo.adb")]:
                raise RuntimeError("Mock evaluation error")
            return EvaluationStatsBuild(
                compiled=True, pre_format_warnings=False, post_format_warnings=False
            )

    def mock_create_eval(evaluation: Eval):
        if evaluation == Eval.BUILD:
            return MockEval()
        return create_eval(evaluation)

    error_msg = (
        "error during baseline evaluation of sample 'bad' in dataset 'spark_check': "
        "EvaluationStatsFailed(eval=<Eval.BUILD: 'build'>, "
        "exception=\"RuntimeError('Mock evaluation error')\")"
    )
    with (
        patch("ada_eval.evals.evaluate.create_eval", mock_create_eval),
        pytest.raises(InvalidBaselineEvaluationError, match=re.escape(error_msg)),
    ):
        run_check()
    assert_log(caplog, INFO, reeval_msg)
    assert_log(caplog, INFO, baseline_msg)
    assert_log(caplog, ERROR, "Error during evaluation of sample bad")
    assert_log(caplog, ERROR, "Error during evaluation of sample good")
    assert len(caplog.records) == 4
    caplog.clear()

    # Test checking an empty list of dataset directories
    check_base_datasets([], jobs=8)
    assert_log(caplog, INFO, "No dataset directories specified; nothing to check.")
    assert len(caplog.records) == 1


def test_check_src_patterns(check_test_datasets: Path):
    """A `src_pattern` may pin a flow aspect, but not a proof rule."""
    datasets = load_datasets(check_test_datasets)
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    sample = spark_dataset.samples[0]

    # A flow aspect has a single correct form, so pinning its text is exact
    sample.required_checks = [
        ProofCheck(rule="DEPENDS_WRONG", src_pattern=r"Depends => \(X => X\)")
    ]
    check_src_patterns(datasets)

    # A proof rule may be required, just not pinned by source text
    sample.required_checks = [ProofCheck(rule="VC_POSTCONDITION")]
    check_src_patterns(datasets)

    sample.required_checks = [ProofCheck(rule="VC_POSTCONDITION", src_pattern="X >= 0")]
    error_msg = (
        "sample 'good' in dataset 'spark_check' has a 'src_pattern' on proof "
        "rule 'VC_POSTCONDITION'."
    )
    with pytest.raises(ProofRuleSrcPatternError, match=re.escape(error_msg)):
        check_src_patterns(datasets)


def test_check_expected_non_spark_entities(check_test_datasets: Path):
    """Require one metadata entry per non-SPARK entity in the canonical result."""
    datasets = load_datasets(check_test_datasets)
    check_expected_non_spark_entities(datasets)
    spark_dataset = next(d for d in datasets if dataset_has_sample_type(d, SparkSample))
    sample = next(s for s in spark_dataset.samples if s.name == "good")
    assert isinstance(sample, SparkSample)

    sample.expected_non_spark_entities = ["Helper"]
    error_msg = (
        "sample 'good' in dataset 'spark_check' declares "
        "'expected_non_spark_entities' ['Helper'], but its canonical solution has []"
    )
    with pytest.raises(UnexpectedNonSparkEntitiesError, match=re.escape(error_msg)):
        check_expected_non_spark_entities(datasets)

    # Overloaded names contribute one entry per declaration.
    stats = next(
        s
        for s in sample.canonical_evaluation_results
        if isinstance(s, EvaluationStatsProve)
    )
    stats.non_spark_entities = ["Helper", "Helper"]
    error_msg = (
        "sample 'good' in dataset 'spark_check' declares "
        "'expected_non_spark_entities' ['Helper'], but its canonical solution has "
        "['Helper', 'Helper']"
    )
    with pytest.raises(UnexpectedNonSparkEntitiesError, match=re.escape(error_msg)):
        check_expected_non_spark_entities(datasets)
    sample.expected_non_spark_entities = ["Helper", "Helper"]
    check_expected_non_spark_entities(datasets)


def lemma_sample(
    lemma: str = "procedure Lemma is\nbegin\n   null;\nend Lemma;\n",
    aspects: str = "",
    **fields: object,
) -> ProofSample:
    """Build a proof sample with an editable ghost procedure."""
    files = DirectoryContents(
        {
            Path("main.gpr"): (
                b'project Main is\n   for Source_Dirs use ("src");\n'
                b'   for Object_Dir use "obj";\n'
                b'   for Create_Missing_Dirs use "True";\nend Main;\n'
            ),
            Path("src/p.ads"): (
                "package P with SPARK_Mode is\n"
                f"   procedure Lemma with Ghost, Global => null{aspects};\nend P;\n"
            ).encode(),
            Path("src/p.adb"): (
                b"package body P with SPARK_Mode is\n"
                b"   procedure Lemma is separate;\nend P;\n"
            ),
            Path("src/p-lemma.adb"): f"separate (P)\n{lemma}".encode(),
        }
    )
    return ProofSample(
        **{
            "name": "lemma",
            "location": Location(path=Path("src/p.ads"), subprogram_name="Lemma"),
            "prompt": "",
            "comments": "",
            "sources": files,
            "canonical_solution": files,
            "canonical_evaluation_results": [],
            "editable_path": Path("src/p-lemma.adb"),
            **fields,
        }
    )


def test_check_editable_files():
    """Check the allowed differences between the base and canonical subunit."""

    def check(sample: ProofSample) -> None:
        dataset = Dataset(name="proof", sample_type=ProofSample, samples=[sample])
        check_editable_files([dataset])

    check(lemma_sample())
    check(
        lemma_sample(
            lemma="procedure Lemma is\n   pragma Loop_Variant (Increases => 1);\n"
        )
    )

    with pytest.raises(EditableFileError, match="declares no 'editable_path'"):
        check(lemma_sample(editable_path=None))
    with pytest.raises(EditableFileError, match="lacks its editable file"):
        check(lemma_sample(editable_path=Path("src/p-other.adb")))
    changed = lemma_sample()
    changed.canonical_solution = DirectoryContents(
        changed.canonical_solution.files | {Path("src/p.ads"): b"package P is end P;"}
    )
    with pytest.raises(
        EditableFileError, match=re.escape("other than 'src/p-lemma.adb'")
    ):
        check(changed)
    with pytest.raises(EditableFileError, match="that is not a subunit"):
        check(lemma_sample(lemma="-- separate (P)\n"))

    # A removable base assertion could invalidate the challenge.
    with pytest.raises(ExistingAssertionError, match="`pragma assert`"):
        check(lemma_sample(lemma="procedure Lemma is begin pragma Assert (True); end;"))

    # Keep trusted declarations outside the file the model can edit.
    check(lemma_sample(expected_non_spark_entities=["P.Other", "P.Lemmas"]))
    with pytest.raises(
        TrustedEntityInEditableFileError, match=re.escape("'P.LEMMA.Stub'")
    ):
        check(lemma_sample(expected_non_spark_entities=["P.LEMMA.Stub"]))

    # A trusted contract is out of a solution's reach only behind an editable file.
    spark = dict(lemma_sample(editable_path=None)) | {
        "unit_tests": DirectoryContents({})
    }
    check(SparkSample(**spark))
    with pytest.raises(EditableFileError, match="declares no 'editable_path'"):
        check(SparkSample(**spark | {"expected_non_spark_entities": ["P.Other"]}))


def test_check_object_dirs():
    """Require the object directory that the proof evaluation reads."""

    def check(gpr: bytes) -> None:
        sample = lemma_sample()
        sample.canonical_solution = DirectoryContents(
            sample.canonical_solution.files | {Path("main.gpr"): gpr}
        )
        dataset = Dataset(name="proof", sample_type=ProofSample, samples=[sample])
        check_object_dirs([dataset])

    gpr = lemma_sample().sources.files[Path("main.gpr")]
    check(gpr.replace(b"for Object_Dir use", b"FOR object_dir\n   use"))
    for bad in (
        gpr.replace(b'"obj"', b'"build"'),
        gpr.replace(b"   for Object_Dir", b"   --  for Object_Dir"),
    ):
        with pytest.raises(ObjectDirError, match="'lemma'"):
            check(bad)


def test_with_infinite_loop():
    """Insert the loop before each legal form of the procedure's end."""

    def looped(lemma: str) -> str:
        solution = with_infinite_loop(lemma_sample(lemma=lemma))
        assert solution is not None
        return solution.files[Path("src/p-lemma.adb")].decode()

    loop = "   loop\n      null;\n   end loop;\n"
    start = "separate (P)\nprocedure Lemma is\nbegin\n   null;\n"
    for end in ("end Lemma;\n", "end;\n", "end Lemma;\n--  Not the end;\n"):
        assert (
            looped(f"procedure Lemma is\nbegin\n   null;\n{end}") == start + loop + end
        )
    assert (
        looped("procedure Lemma is begin null; end;\n")
        == f"separate (P)\nprocedure Lemma is begin null; {loop}end;\n"
    )
    nested = (
        "procedure Lemma is\n   procedure Q is\n   begin\n      null;\n   end Q;\n"
        "begin\n   Q;\n"
    )
    assert looped(nested + "end Lemma;\n").endswith(f"   Q;\n{loop}end Lemma;\n")
    assert (
        with_infinite_loop(lemma_sample(lemma="function Lemma return Boolean;")) is None
    )


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("gnatprove"), reason="gnatprove not available")
@pytest.mark.skipif(not shutil.which("gprbuild"), reason="gprbuild not available")
@pytest.mark.skipif(not shutil.which("gprls"), reason="gprls not available")
def test_check_editable_procedures_terminate():
    """Require the added loop to fail proof or to change the object code."""

    def check(sample: ProofSample) -> None:
        dataset = Dataset(name="proof", sample_type=ProofSample, samples=[sample])
        check_editable_procedures_terminate([dataset], jobs=1)

    # Proof must reject the infinite loop even though ghost code is erased.
    with pytest.raises(NonTerminatingSolutionPassedError, match="'lemma'"):
        check(lemma_sample())
    check(lemma_sample(aspects=", Always_Terminates"))

    # A loop in executable code changes the object file instead.
    plain = lemma_sample()
    spec = plain.sources.files[Path("src/p.ads")].replace(b"with Ghost, ", b"with ")
    plain.sources = plain.canonical_solution = DirectoryContents(
        plain.sources.files | {Path("src/p.ads"): spec}
    )
    check(plain)

    # GNATprove rejects a ghost call there that might not return, so erased
    # code cannot hide a loop in an executable procedure either.
    spinner = (
        b"separate (P)\nprocedure Lemma is\n"
        b"   procedure Spin with Ghost, Post => False is\n   begin\n"
        b"      loop\n         null;\n      end loop;\n   end Spin;\n"
        b"begin\n   Spin;\nend Lemma;\n"
    )
    plain.canonical_solution = DirectoryContents(
        plain.sources.files | {Path("src/p-lemma.adb"): spinner}
    )
    dataset = Dataset(name="proof", sample_type=ProofSample, samples=[plain])
    (generated,) = generate_canonical([dataset])[0].samples
    assert isinstance(generated, GeneratedProofSample)
    assert Prove().evaluate(generated).result == "unproved"
