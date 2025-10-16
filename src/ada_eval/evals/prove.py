import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from ada_eval.datasets import (
    Eval,
    EvaluatedSparkSample,
    EvaluationStatsProve,
    GeneratedSparkSample,
    ProofCheck,
    SubprogramNotFoundError,
)
from ada_eval.utils import (
    check_on_path,
    index_from_line_and_col,
    run_cmd_with_timeout,
    sort_dict,
    type_checked,
)

from .generic_eval import GenericEval

logger = logging.getLogger(__name__)


PROVE_TIMEOUT_S = 60


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
        proof_steps=0,
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
    proof_steps: int


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
        for proof_result_unchecked in type_checked(spark_data[check_kind], list):
            proof_result = type_checked(proof_result_unchecked, dict)
            proof_steps = (
                sum(
                    type_checked(type_checked(info, dict)["steps"], int)
                    for node in type_checked(proof_result["check_tree"], list)
                    for info in type_checked(node["proof_attempts"], dict).values()
                )
                if check_kind == "proof"
                else 0
            )
            results.append(
                ProofResult(
                    entity_name=entity_names[type_checked(proof_result["entity"], int)],
                    rule=type_checked(proof_result["rule"], str),
                    severity=type_checked(proof_result["severity"], str),
                    file=type_checked(proof_result["file"], str),
                    line=type_checked(proof_result["line"], int),
                    col=type_checked(proof_result["col"], int),
                    proof_steps=proof_steps,
                )
            )
    return results


class ProofResultSourceNotFoundError(ValueError):
    """Raised if a source file reported by GNATprove cannot be unambiguously located."""


def proof_check_is_satisfied(
    check: ProofCheck, proof_result: ProofResult, sample: GeneratedSparkSample
) -> bool:
    """
    Check that `proof_result` describes a successful proof of a check matching `check`.

    Args:
        check: The `ProofCheck` to check against
        proof_result: The `ProofResult` parsed from the JSON in GNATprove's
            `.spark` file
        sample: The `GeneratedSparkSample` being evaluated (used to find the
            source files and subprogram of interest for matching against
            `check.src_pattern`)

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
    src_files = {
        p for p in sample.generated_solution.files if p.name == proof_result.file
    }
    if len(src_files) == 0:
        # All source files reported by GNATprove should be somewhere in the
        # generated solution
        msg = (
            f"file '{proof_result.file}' reported by GNATprove, but not found "
            f"in generated solution for sample '{sample.name}'."
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
        with sample.generated_solution.unpacked() as working_dir:
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
            )
        # The output should be a newline-separated list of the absolute
        # paths of all source files in the project, with some whitespace and
        # potentially duplicates.
        src_files = {
            Path(line.strip())
            for line in gprls_result.stdout.split("\n")
            if line.strip() != ""
        }
        src_files = sample.generated_solution.files.keys() & {
            p.relative_to(working_dir) for p in src_files if p.name == proof_result.file
        }
        if len(src_files) != 1:
            # Ada source file basenames should be unique within a project.
            msg = (
                f"expected exactly 1 '{proof_result.file}' file in generated "
                f"solution of sample '{sample.name}', but gprls found {len(src_files)}."
            )
            raise ProofResultSourceNotFoundError(msg)
    src_file = next(iter(src_files))
    src_str = sample.generated_solution.files[src_file]
    # We only want to match starting from the check location reported by
    # GNATprove
    src_str = src_str[
        index_from_line_and_col(src_str, proof_result.line, proof_result.col) :
    ]
    # Check if the `src_pattern` matches
    return re.match(check.src_pattern, src_str) is not None


class Prove(GenericEval[GeneratedSparkSample, EvaluatedSparkSample]):
    """An evaluation that runs GNATprove and checks for any proof failures."""

    eval: ClassVar = Eval.PROVE
    supported_types: ClassVar = {GeneratedSparkSample: EvaluatedSparkSample}

    def __init__(self) -> None:
        # Detect missing `gnatprove` before attempting any evaluation for a
        # cleaner error output.
        check_on_path("gnatprove")
        check_on_path("gprls")

    def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsProve:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug(
                "Evaluating '%s' with GNATprove in %s", sample.name, working_dir
            )
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
            # Run `gnatprove` on the unit specified by `sample.location`.
            proc_result, _ = run_cmd_with_timeout(
                ["gnatprove", "-Pmain.gpr", "-k", str(sample.location.path)],
                working_dir,
                PROVE_TIMEOUT_S,
            )
            if proc_result.returncode != 0:
                return empty_prove_stats("error")
            # Parse the `.spark` JSON files
            gnatprove_dir = working_dir / "obj" / "gnatprove"
            spark_files = [
                type_checked(json.loads(spark_file_path.read_text("utf-8")), dict)
                for spark_file_path in gnatprove_dir.glob("*.spark")
            ]
        # Extract the `proof_result`s from the `.spark` files
        proof_results = [
            r for spark_file in spark_files for r in extract_proof_results(spark_file)
        ]
        # Tally up the checks and warnings by rule
        rule_counts = {
            severity: Counter[str]()
            for severity in ("info", "low", "medium", "high", "warning", "error")
        }
        for proof_result in proof_results:
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
        # Find any entities for which SPARK was disabled
        non_spark_entities: list[str] = []
        for spark_file in spark_files:
            entity_names = extract_entities(spark_file)
            non_spark_entities.extend(
                entity_names[int(num_str)]
                for num_str, value in type_checked(spark_file["spark"], dict).items()
                if value != "all"
            )
        # Determine which `required_checks` have been satisfied
        missing_proof_checks = list(sample.required_checks)
        for proof_result in proof_results:
            for idx, proof_check in enumerate(missing_proof_checks):
                if proof_check_is_satisfied(proof_check, proof_result, sample):
                    missing_proof_checks.pop(idx)
                    break
        # Count the number of `pragma Assume`s
        pragma_assume_count = sum(
            len(type_checked(spark_file["pragma_assume"], list))
            for spark_file in spark_files
        )
        # Sum the number of proof steps
        total_proof_steps = sum(
            proof_result.proof_steps for proof_result in proof_results
        )
        # Categorise the overall result
        if unproved_checks.total() > 0:
            result = "unproved"
        elif not (
            len(non_spark_entities) == 0
            and len(missing_proof_checks) == 0
            and warnings.total() == 0
            and pragma_assume_count == 0
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
            proof_steps=total_proof_steps,
        )
