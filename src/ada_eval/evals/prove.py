import json
import logging
import re
from collections import Counter
from typing import ClassVar, Literal

from pydantic import BaseModel

from ada_eval.datasets import (
    Eval,
    EvaluatedSparkSample,
    EvaluationStatsProve_New,
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
) -> EvaluationStatsProve_New:
    """Return a new `EvaluationStatsProve` which is empty apart from the `result`."""
    return EvaluationStatsProve_New(
        result=result,
        proved_checks=Counter(),
        unproved_checks=Counter(),
        warnings=Counter(),
        missing_required_checks=0,
        pragma_assume_count=0,
        proof_steps=0,
    )


class ProofResult(BaseModel):
    """A single `proof_result` entry from a GNATprove `.spark` file."""

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
        KeyError or UnexpectedTypeError: if `spark_data` has an unexpected
            structure

    """
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
                    rule=type_checked(proof_result["rule"], str),
                    severity=type_checked(proof_result["severity"], str),
                    file=type_checked(proof_result["file"], str),
                    line=type_checked(proof_result["line"], int),
                    col=type_checked(proof_result["col"], int),
                    proof_steps=proof_steps,
                )
            )
    return results


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
    if proof_result.rule != check.rule or proof_result.severity != "info":
        # Not the right rule or not successfully proven
        return False
    if check.src_pattern is None:
        # Successfully proven, and no source pattern to check
        return True
    # The check may occur in either the body or spec file
    source_files = sample.generated_solution.files.keys() & {
        sample.location.path.with_suffix(".ads"),
        sample.location.path.with_suffix(".adb"),
    }
    # Get the contents of the source file of interest
    src_file_path = next(p for p in source_files if p.name == proof_result.file)
    src_str = sample.generated_solution.files[src_file_path]
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

    def evaluate(self, sample: GeneratedSparkSample) -> EvaluationStatsProve_New:
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug(
                "Evaluating '%s' with GNATprove in %s", sample.name, working_dir
            )
            # Search for the subprogram of interest.
            try:
                sample.location.find_line_number(working_dir)
            except SubprogramNotFoundError:
                return empty_prove_stats("subprogram_not_found")
            # Run `gnatprove` on the unit and subprogram specified by
            # `sample.location`.
            proc_result, _ = run_cmd_with_timeout(
                [
                    "gnatprove",
                    "-Pmain.gpr",
                    "-k",
                    str(sample.location.path),
                ],
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
            pragma_assume_count == len(missing_proof_checks) == warnings.total() == 0
        ):
            result = "proved_incorrectly"
        else:
            result = "proved"
        # Return the `EvaluationStats` (sorting counters by key for stable output)
        return EvaluationStatsProve_New(
            result=result,
            proved_checks=Counter(sort_dict(proved_checks)),
            unproved_checks=Counter(sort_dict(unproved_checks)),
            warnings=Counter(sort_dict(warnings)),
            missing_required_checks=len(missing_proof_checks),
            pragma_assume_count=pragma_assume_count,
            proof_steps=total_proof_steps,
        )
