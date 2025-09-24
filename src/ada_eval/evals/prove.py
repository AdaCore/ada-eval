import json
import logging
import re
from collections import Counter
from typing import ClassVar, Literal

from ada_eval.datasets import (
    Eval,
    EvaluatedSparkSample,
    EvaluationStatsProve_New,
    GeneratedSparkSample,
    ProofCheck,
    SubprogramNotFoundError,
)
from ada_eval.utils import check_on_path, run_cmd_with_timeout, sort_dict, type_checked

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


def proof_check_is_satisfied(
    check: ProofCheck, proof_result: dict[object, object], sample: GeneratedSparkSample
) -> bool:
    """
    Check that `proof_result` describes a successful proof of a check matching `check`.

    Args:
        check: The `ProofCheck` to check against
        proof_result: The `proof_result` parsed from the JSON in GNATprove's
            `.spark` file
        sample: The `GeneratedSparkSample` being evaluated (used to find the
            source files and subprogram of interest for matching against
            `check.src_pattern`)

    Raises:
        KeyError or UnexpectedTypeError: if `proof_result` has an unexpected
            structure

    """
    if proof_result["rule"] != check.rule or proof_result["severity"] != "info":
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
    src_file_path = next(p for p in source_files if p.name == proof_result["file"])
    src_str = sample.generated_solution.files[src_file_path]
    # We only want to match starting from the check location reported by
    # GNATprove
    line = type_checked(proof_result["line"], int)
    column = type_checked(proof_result["col"], int)
    src_str = "".join(src_str.splitlines(keepends=True)[line - 1 :])[column - 1 :]
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
                subp_lineno = sample.location.find_line_number(working_dir)
            except SubprogramNotFoundError:
                return empty_prove_stats("subprogram_not_found")
            # Run `gnatprove` on the unit and subprogram specified by
            # `sample.location`.
            proc_result, _ = run_cmd_with_timeout(
                [
                    "gnatprove",
                    "-Pmain.gpr",
                    "-k",
                    f"--limit-subp={sample.location.path.name}:{subp_lineno}",
                    str(sample.location.path),
                ],
                working_dir,
                PROVE_TIMEOUT_S,
            )
            if proc_result.returncode != 0:
                return empty_prove_stats("error")
            # Parse the `.spark` JSON file
            gnatprove_dir = working_dir / "obj" / "gnatprove"
            spark_file_path = gnatprove_dir / (sample.location.path.stem + ".spark")
            spark_data = type_checked(
                json.loads(spark_file_path.read_text("utf-8")), dict
            )
        # Process each `proof_result` in the `.spark` file.
        check_results = {
            severity: Counter[str]()
            for severity in ("info", "low", "medium", "high", "warning", "error")
        }
        missing_proof_checks = list(sample.required_checks)
        total_proof_steps = 0
        for check_kind in ("flow", "proof"):
            for proof_result_unchecked in type_checked(spark_data[check_kind], list):
                proof_result = type_checked(proof_result_unchecked, dict)
                # Record the rule and severity.
                check_results[proof_result["severity"]][proof_result["rule"]] += 1
                # `pop()` any `required_check` that it satisfies.
                for idx, proof_check in enumerate(missing_proof_checks):
                    if proof_check_is_satisfied(proof_check, proof_result, sample):
                        missing_proof_checks.pop(idx)
                        break
                # Add the number of proof steps to the total.
                if check_kind == "proof":
                    total_proof_steps += sum(
                        type_checked(type_checked(info, dict)["steps"], int)
                        for node in type_checked(proof_result["check_tree"], list)
                        for info in type_checked(node["proof_attempts"], dict).values()
                    )
        proved_checks = check_results["info"]
        unproved_checks = (
            check_results["low"] + check_results["medium"] + check_results["high"]
        )
        warnings = check_results["warning"]
        # Count the `pragma Assume`s
        pragma_assume_count = len(spark_data["pragma_assume"])
        # Treat any errors as an overall error (though it is not clear if a
        # `.spark` file will ever actually be created if there are errors).
        if check_results["error"].total() > 0:
            return empty_prove_stats("error")
        # Return the `EvaluationStats`
        if (unproved_checks + warnings).total() > 0:
            result = "unproved"
        elif pragma_assume_count > 0 or len(missing_proof_checks) > 0:
            result = "proved_incorrectly"
        else:
            result = "proved"
        # Sort counters by key for stable output
        return EvaluationStatsProve_New(
            result=result,
            proved_checks=Counter(sort_dict(proved_checks)),
            unproved_checks=Counter(sort_dict(unproved_checks)),
            warnings=Counter(sort_dict(warnings)),
            missing_required_checks=len(missing_proof_checks),
            pragma_assume_count=pragma_assume_count,
            proof_steps=total_proof_steps,
        )
