import json
import logging
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Container, Iterable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from ada_eval.datasets import (
    Eval,
    EvaluatedProofSample,
    EvaluatedSparkSample,
    EvaluationStatsProve,
    GeneratedProofSample,
    GeneratedSparkSample,
    HarnessResult,
    ProofCheck,
    SubprogramNotFoundError,
)
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.utils import (
    check_on_path,
    gpr_tools_env,
    index_from_line_and_col,
    run_cmd_with_timeout,
    sort_dict,
    type_checked,
)

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


PROVE_TIMEOUT_S = 300

PROOF_HARNESS_PROJECT = "harness.gpr"

GNATPROVE_OUTPUT_TAIL_LINES = 40


def log_gnatprove_failure(
    sample_name: str, proc_result: subprocess.CompletedProcess[str]
) -> None:
    """Log the GNATprove command and its last lines of output on failure."""
    output = (proc_result.stdout + proc_result.stderr).splitlines()
    logger.warning(
        "GNATprove failed with exit code %d on sample '%s': %s\n%s",
        proc_result.returncode,
        sample_name,
        " ".join(str(arg) for arg in proc_result.args),
        "\n".join(output[-GNATPROVE_OUTPUT_TAIL_LINES:]),
    )


def empty_prove_stats(
    result: Literal["subprogram_not_found", "error"],
) -> EvaluationStatsProve:
    """Return a new `EvaluationStatsProve` which is empty apart from the `result`."""
    return EvaluationStatsProve(
        result=result,
        proved_checks=Counter(),
        unproved_checks=Counter(),
        warnings=Counter(),
        non_spark_entities=[],
        missing_required_checks=[],
        pragma_assume_count=0,
    )


def extract_entities(spark_data: dict[str, object]) -> dict[int, str]:
    """
    Extract an entity number-to-name mapping from a GNATprove `.spark` file's JSON data.

    Raises:
        KeyError, UnexpectedTypeError or ValueError: if `spark_data` has an
            unexpected structure

    """
    return {
        int(entity_num_str): type_checked(entity["name"], str)
        for entity_num_str, entity in type_checked(spark_data["entities"], dict).items()
    }


class ProofResult(BaseModel):
    """A single `proof_result` entry from a GNATprove `.spark` file."""

    entity_name: str
    rule: str
    severity: Literal["info", "low", "medium", "high", "warning", "error"]
    file: str
    line: int
    col: int
    justified: bool = False  # The submitted code justifies this check.


def extract_proof_results(spark_data: dict[str, object]) -> list[ProofResult]:
    """
    Extract a list of `ProofResult`s from a GNATprove `.spark` file's JSON data.

    Raises:
        KeyError, UnexpectedTypeError or ValueError: if `spark_data` has an
            unexpected structure

    """
    entity_names = extract_entities(spark_data)
    results: list[ProofResult] = []
    for check_kind in ("flow", "proof"):
        # GNATprove omits result sections for units it did not analyse.
        for proof_result_unchecked in type_checked(
            spark_data.get(check_kind, []), list
        ):
            proof_result = type_checked(proof_result_unchecked, dict)
            results.append(
                ProofResult(
                    entity_name=entity_names[type_checked(proof_result["entity"], int)],
                    rule=type_checked(proof_result["rule"], str),
                    severity=type_checked(proof_result["severity"], str),
                    file=type_checked(proof_result["file"], str),
                    line=type_checked(proof_result["line"], int),
                    col=type_checked(proof_result["col"], int),
                    justified="suppressed" in proof_result,
                )
            )
    return results


def extract_entity_files(spark_data: dict[str, object]) -> dict[int, str]:
    """
    Map GNATprove entity numbers to the basenames of their declaring files.

    Include entities for which GNATprove reports a source location.

    Raises:
        KeyError, UnexpectedTypeError or ValueError: if `spark_data` has an
            unexpected structure

    """
    files: dict[int, str] = {}
    for entity_num_str, entity_unchecked in type_checked(
        spark_data["entities"], dict
    ).items():
        slocs = type_checked(entity_unchecked, dict).get("sloc")
        if isinstance(slocs, list) and len(slocs) > 0:
            sloc = type_checked(slocs[0], dict)
            files[int(entity_num_str)] = Path(type_checked(sloc["file"], str)).name
    return files


# GNAT embeds the source location in the wrapper name for a generic instance.
# P.Freegp603.Free and P.FREEGP603.Free both denote P.Free.
INSTANCE_WRAPPER = re.compile(r"\b(\w+)gp\d+\.(\1)\b", re.IGNORECASE)


def extract_non_spark_entities(
    spark_files: Sequence[dict[str, object]], declared_in: Container[str] | None = None
) -> list[str]:
    """
    List the entities that GNATprove places outside full SPARK analysis.

    Strip the location-dependent wrapper from generic instance names so a
    source edit above the declaration still yields the same name.

    Args:
        spark_files: The JSON data of GNATprove's `.spark` files
        declared_in: If given, only entities declared in a source file with one
            of these names are returned. An entity whose declaration GNATprove
            does not locate is returned regardless.

    """
    return [
        name
        for spark_file in spark_files
        for name in _entity_names(
            spark_file,
            [
                int(num_str)
                for num_str, value in type_checked(spark_file["spark"], dict).items()
                if value != "all"
            ],
            declared_in,
        )
    ]


def extract_skipped_entities(
    spark_files: Sequence[dict[str, object]], declared_in: Container[str] | None = None
) -> list[str]:
    """
    List outermost entities excluded from proof by a GNATprove annotation.

    One entry covers all skipped entities nested within it.

    Args:
        spark_files: The JSON data of GNATprove's `.spark` files
        declared_in: As for `extract_non_spark_entities`.

    """
    skipped = {
        name
        for spark_file in spark_files
        for name in _entity_names(
            spark_file,
            [
                type_checked(num, int)
                for key in ("skip_proof", "skip_flow_proof")
                for num in type_checked(spark_file.get(key, []), list)
            ],
            declared_in,
        )
    }
    return sorted(
        name
        for name in skipped
        if not any(name.startswith(outer + ".") for outer in skipped)
    )


def _entity_names(
    spark_file: dict[str, object],
    entity_nums: Iterable[int],
    declared_in: Container[str] | None,
) -> list[str]:
    """Resolve entity numbers to names and apply the source-file filter."""
    entity_names = extract_entities(spark_file)
    entity_files = extract_entity_files(spark_file)
    return [
        INSTANCE_WRAPPER.sub(r"\2", entity_names[num])
        for num in entity_nums
        if declared_in is None
        or num not in entity_files
        or entity_files[num] in declared_in
    ]


def count_pragma_assumes(spark_files: Sequence[dict[str, object]]) -> int:
    """Count `pragma Assume` uses reported by GNATprove."""
    return sum(
        len(type_checked(spark_file.get("pragma_assume", []), list))
        for spark_file in spark_files
    )


ADA_TOKEN = re.compile(r'"(?:[^"\n]|"")*"|\'.\'(?!\w)|\w+|\S')
ADA_COMMENT = re.compile(r'("(?:[^"\n]|"")*")|--[^\n]*')


def ada_tokens(text: str) -> list[str]:
    """Tokenize Ada source, preserving literals and folding identifier case."""
    text = ADA_COMMENT.sub(lambda match: match.group(1) or "", text)
    return [
        tok if tok.startswith(('"', "'")) else tok.casefold()
        for tok in ADA_TOKEN.findall(text)
    ]


class ProofResultSourceNotFoundError(ValueError):
    """Raised if a source file reported by GNATprove cannot be unambiguously located."""


def proof_check_is_satisfied(
    check: ProofCheck,
    proof_result: ProofResult,
    sample: GeneratedProofSample,
    solution: DirectoryContents,
) -> bool:
    """
    Return whether GNATprove discharged the required rule for the requested entity.

    When the check specifies a source pattern, match it at the line and column
    GNATprove reported. Use the project to resolve ambiguous source filenames.

    Raises:
        ProofResultSourceNotFoundError: The reported filename cannot be resolved
            to a single project source.

    """
    if (
        proof_result.rule != check.rule
        or (
            check.entity_name is not None
            and proof_result.entity_name != check.entity_name
        )
        or proof_result.severity != "info"
    ):
        # Not the right rule and entity name, or not successfully proven
        return False
    if check.src_pattern is None:
        # Successfully proven, and no source pattern to check
        return True
    # Find the contents of the source file corresponding to `proof_result.file`
    # (which specifies only the basename).
    src_files = {p for p in solution.files if p.name == proof_result.file}
    if len(src_files) == 0:
        msg = (
            f"GNATprove reported '{proof_result.file}', which is missing from "
            f"the evaluated project for sample '{sample.name}'."
        )
        raise ProofResultSourceNotFoundError(msg)
    if len(src_files) != 1:
        # In the (hopefully unlikely) event that there are multiple files with
        # the same basename, use `gprls` to find which one is actually part of
        # the project.
        logger.debug(
            "Multiple files found for '%s' in sample '%s'; disambiguating with gprls.",
            proof_result.file,
            sample.name,
        )
        with solution.unpacked() as working_dir:
            # Community versions of `gprls` require that the project is built first.
            run_cmd_with_timeout(
                ["gprbuild", "-Pmain.gpr"],
                working_dir,
                PROVE_TIMEOUT_S,
                check=True,
                env=gpr_tools_env(),
            )
            gprls_result, _ = run_cmd_with_timeout(
                [
                    "gprls",
                    "-Pmain.gpr",
                    "-s",  # Print source file paths (not object files)
                    "-d",  # Print dependencies (i.e. include `.ads` files)
                    "-U",  # Include `with`ed projects
                    # No positional args implicitly specifies all `.adb` files
                ],
                working_dir,
                PROVE_TIMEOUT_S,
                check=True,
                env=gpr_tools_env(),
            )
        # The output should be a newline-separated list of the absolute
        # paths of all source files in the project, with some whitespace and
        # potentially duplicates.
        src_files = {
            Path(line.strip())
            for line in gprls_result.stdout.split("\n")
            if line.strip() != ""
        }
        src_files = solution.files.keys() & {
            p.relative_to(working_dir) for p in src_files if p.name == proof_result.file
        }
        if len(src_files) != 1:
            # Ada source file basenames should be unique within a project.
            msg = (
                f"Expected one source named '{proof_result.file}' in the evaluated "
                f"project for sample '{sample.name}'. gprls found {len(src_files)}."
            )
            raise ProofResultSourceNotFoundError(msg)
    src_file = next(iter(src_files))
    src_str = solution.files[src_file].decode("utf-8")
    # We only want to match starting from the check location reported by
    # GNATprove
    src_str = src_str[
        index_from_line_and_col(src_str, proof_result.line, proof_result.col) :
    ]
    # Check if the `src_pattern` matches
    return re.match(check.src_pattern, src_str) is not None


def evaluate_proof_harness(
    sample: GeneratedProofSample, solution_dir: Path
) -> HarnessResult:
    """
    Run GNATprove on `sample`'s proof harness against its generated solution.

    The harness wraps the subprogram of interest in a subprogram with the same
    inputs and outputs, whose postcondition states the properties the
    solution's contracts must imply. GNATprove analyzes a caller using only the
    callee's contract, so the harness proves when those contracts are at least
    as strong as the sample requires.

    Args:
        sample: The sample being evaluated. Must have a non-empty `proof_harness`.
        solution_dir: The directory the generated solution is unpacked in,
            which the harness project is resolved against.

    """
    env = gpr_tools_env(solution_dir)
    with sample.proof_harness.unpacked() as harness_dir:
        logger.debug("Proving harness for '%s' in %s", sample.name, harness_dir)
        harness_sources = sorted(
            path.name
            for path in sample.proof_harness.files
            if path.suffix in (".ads", ".adb")
        )
        proc_result, _ = run_cmd_with_timeout(
            [
                "gnatprove",
                f"-P{PROOF_HARNESS_PROJECT}",
                "--warnings=error",
                "-u",
                *harness_sources,
            ],
            harness_dir,
            PROVE_TIMEOUT_S,
            env=env,
        )
        if proc_result.returncode != 0:
            log_gnatprove_failure(sample.name, proc_result)
            return "error"
        spark_files = [
            type_checked(json.loads(spark_file_path.read_text("utf-8")), dict)
            for spark_file_path in (harness_dir / "obj" / "gnatprove").glob("*.spark")
        ]
    if len(spark_files) == 0:
        return "error"
    proof_results = [
        result
        for spark_file in spark_files
        for result in extract_proof_results(spark_file)
    ]
    wrapper_name = f"Proof_Harness.Check_{sample.location.subprogram_name}".casefold()
    if (
        not any(
            result.rule == "VC_POSTCONDITION"
            and result.entity_name.casefold() == wrapper_name
            for result in proof_results
        )
        or count_pragma_assumes(spark_files) > 0
        or len(extract_non_spark_entities(spark_files)) > 0
        or len(extract_skipped_entities(spark_files)) > 0
        or any(result.justified for result in proof_results)
    ):
        return "error"
    return (
        "unproved"
        if any(
            result.severity != "info"
            for result in proof_results
            if result.severity != "warning"
        )
        else "proved"
    )


EXECUTABLE_CODE_PRAGMAS = (
    "pragma Assertion_Policy (Disable);\n"
    "pragma Assertion_Policy (Ghost => Ignore);\n"
    "pragma Suppress_Exception_Locations;\n"
    + "".join(
        f"pragma Ignore_Pragma ({name});\n"
        for name in (
            "Assert",
            "Assert_And_Cut",
            "Assume",
            "Check",
            "Loop_Invariant",
            "Loop_Variant",
        )
    )
)
"""
Compiler configuration for comparing executable code across proof edits.

Disabling contracts and assertions avoids emitting finalizers or secondary
stack operations. Ignore assertion pragmas explicitly for levels such as
`Static` with their own policies. Ignoring ghost code suppresses its object
code. Omitting exception locations makes the comparison insensitive to line
number changes.
"""


class NoObjectFilesError(ValueError):
    """The base project produced no objects for the executable-code comparison."""


def compile_objects(solution: DirectoryContents) -> dict[str, bytes] | None:
    """
    Build the project's units with proof material disabled or ignored.

    Returns:
        Object contents keyed by filename, or `None` on compilation failure.

    """
    with solution.unpacked() as working_dir:
        config = working_dir / "ada_eval_executable_code.adc"
        config.write_text(EXECUTABLE_CODE_PRAGMAS)
        proc_result, _ = run_cmd_with_timeout(
            [
                *("gprbuild", "-q", "-Pmain.gpr", "-c", "-u"),
                *("-cargs", "-O0", "-g0", f"-gnatec={config}"),
            ],
            working_dir,
            PROVE_TIMEOUT_S,
            env=gpr_tools_env(),
        )
        if proc_result.returncode != 0:
            return None
        objects = {}
        for path in working_dir.glob("obj/*.o"):
            # Edits can renumber GNAT's local symbols, including finalizers.
            run_cmd_with_timeout(
                ["objcopy", "--discard-all", str(path)],
                working_dir,
                PROVE_TIMEOUT_S,
                check=True,
            )
            objects[path.name] = path.read_bytes()
        return objects


def changed_object_files(sample: GeneratedProofSample) -> list[str]:
    """
    Identify object files affected by edits to the submitted subunit.

    Compile each version of the editable file in the same reference project.
    Compare all project objects to cover generic instantiations. Accept a
    match against either the base or the canonical solution.

    Optimisation could discard an entire unused computation. At -O0, even a
    placeholder `null` can emit an instruction. A submission may keep that
    placeholder or replace it with the reference proof material.

    Raises:
        NoObjectFilesError: If the base sources compile to no object files.

    """
    base = compile_objects(sample.with_editable_file_from(sample.sources))
    if not base:
        msg = f"the base sources of sample '{sample.name}' compile to no object files"
        raise NoObjectFilesError(msg)
    solution = compile_objects(sample.solution_to_evaluate())
    if solution is None:
        return ["(the solution does not compile)"]
    changed = sorted(
        name
        for name in base.keys() | solution.keys()
        if base.get(name) != solution.get(name)
    )
    if len(changed) > 0 and compile_objects(sample.canonical_solution) == solution:
        return []
    return changed


class Prove(GenericEval[GeneratedProofSample, EvaluatedProofSample]):
    """An evaluation that runs GNATprove and checks for any proof failures."""

    eval: ClassVar = Eval.PROVE
    supported_types: ClassVar = {
        GeneratedProofSample: EvaluatedProofSample,
        GeneratedSparkSample: EvaluatedSparkSample,
    }

    def __init__(self) -> None:
        # Detect missing `gnatprove` before attempting any evaluation for a
        # cleaner error output.
        check_on_path("gnatprove")
        check_on_path("gprbuild")
        check_on_path("gprls")
        check_on_path("objcopy")

    def evaluate(self, sample: GeneratedProofSample) -> EvaluationStatsProve:
        solution = sample.solution_to_evaluate()
        stats = self._prove(sample, solution)
        # A passing proof must also preserve the reference executable code.
        if sample.editable_path is not None and stats.result == "proved":
            stats.changed_object_files = changed_object_files(sample)
            if stats.changed_object_files:
                stats.result = "proved_incorrectly"
        # Record submitted project edits even though the reference settings ran.
        stats.changed_project_files = sample.changed_project_files()
        return stats

    def _prove(
        self, sample: GeneratedProofSample, solution: DirectoryContents
    ) -> EvaluationStatsProve:
        """Evaluate the supplied project against the sample's proof requirements."""
        with solution.unpacked() as working_dir:
            logger.debug(
                "Evaluating '%s' with GNATprove in %s", sample.name, working_dir
            )
            # Discard artifacts from earlier runs before collecting proof results.
            shutil.rmtree(working_dir / "obj", ignore_errors=True)
            # Search for the subprogram of interest.
            #
            # We no longer use the line number, but it is still useful to fail
            # fast if the subprogram of interest is completely missing (there
            # are potentially edge cases with nested subprograms etc. though,
            # so the real test for absence is the unit test evaluation).
            try:
                sample.location.find_line_number(working_dir)
            except SubprogramNotFoundError:
                return empty_prove_stats("subprogram_not_found")
            # Generic bodies need their instantiations in the proof scope.
            units = [] if sample.prove_whole_project else [sample.location.path.name]
            proc_result, _ = run_cmd_with_timeout(
                ["gnatprove", "-Pmain.gpr", "-k", *units],
                working_dir,
                PROVE_TIMEOUT_S,
                env=gpr_tools_env(),
            )
            if proc_result.returncode != 0:
                log_gnatprove_failure(sample.name, proc_result)
                return empty_prove_stats("error")
            # Parse the `.spark` JSON files
            gnatprove_dir = working_dir / "obj" / "gnatprove"
            spark_files = [
                type_checked(json.loads(spark_file_path.read_text("utf-8")), dict)
                for spark_file_path in gnatprove_dir.glob("*.spark")
            ]
            harness_result: HarnessResult = (
                evaluate_proof_harness(sample, working_dir)
                if len(sample.proof_harness.files) > 0
                else "not_applicable"
            )
        # Extract the `proof_result`s from the `.spark` files
        proof_results = [
            r for spark_file in spark_files for r in extract_proof_results(spark_file)
        ]
        # Tally up the checks and warnings by rule
        rule_counts = {
            severity: Counter[str]()
            for severity in ("info", "low", "medium", "high", "warning", "error")
        }
        justified_checks = Counter[str]()
        for proof_result in proof_results:
            if proof_result.justified:
                justified_checks[proof_result.rule] += 1
            else:
                rule_counts[proof_result.severity][proof_result.rule] += 1
        proved_checks = rule_counts["info"]
        unproved_checks = (
            rule_counts["low"] + rule_counts["medium"] + rule_counts["high"]
        )
        warnings = rule_counts["warning"]
        # Treat any errors as an overall error (though it is not clear if a
        # `.spark` file will ever actually be created if there are errors).
        if rule_counts["error"].total() > 0:
            return empty_prove_stats("error")
        # Compare the sample's non-SPARK entities with its declared exceptions.
        non_spark_entities = extract_non_spark_entities(
            spark_files, declared_in={p.name for p in solution.files}
        )
        # Ada identifier matching is case-insensitive, including overloads.
        unexpected_non_spark_entities = Counter(
            name.casefold() for name in non_spark_entities
        ) - Counter(name.casefold() for name in sample.expected_non_spark_entities)
        # Restrict skipped-proof diagnostics to the sample's sources.
        skipped_entities = extract_skipped_entities(
            spark_files, declared_in={p.name for p in solution.files}
        )
        # Determine which `required_checks` have been satisfied
        missing_proof_checks = list(sample.required_checks)
        for proof_result in proof_results:
            for idx, proof_check in enumerate(missing_proof_checks):
                if proof_check_is_satisfied(
                    proof_check, proof_result, sample, solution
                ):
                    missing_proof_checks.pop(idx)
                    break
        # Count the number of `pragma Assume`s
        pragma_assume_count = count_pragma_assumes(spark_files)
        # Categorise the overall result
        if unproved_checks.total() > 0:
            result = "unproved"
        elif not (
            len(unexpected_non_spark_entities) == 0
            and len(skipped_entities) == 0
            and justified_checks.total() == 0
            and len(missing_proof_checks) == 0
            and warnings.total() == 0
            and pragma_assume_count == 0
            and harness_result in ("not_applicable", "proved")
        ):
            result = "proved_incorrectly"
        else:
            result = "proved"
        # Return the `EvaluationStats` (sorting for stable output)
        return EvaluationStatsProve(
            result=result,
            proved_checks=Counter(sort_dict(proved_checks)),
            unproved_checks=Counter(sort_dict(unproved_checks)),
            warnings=Counter(sort_dict(warnings)),
            non_spark_entities=sorted(non_spark_entities),
            missing_required_checks=missing_proof_checks,
            pragma_assume_count=pragma_assume_count,
            justified_checks=Counter(sort_dict(justified_checks)),
            skipped_entities=skipped_entities,
            harness=harness_result,
        )
