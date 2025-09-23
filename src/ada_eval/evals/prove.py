import json
import logging
from collections import Counter
from typing import ClassVar, Literal

from ada_eval.datasets import (
    Eval,
    EvaluatedSparkSample,
    EvaluationStatsProve_New,
    GeneratedSparkSample,
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
        pragma_assume_count=0,
        proof_steps=0,
    )


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
            # Run `gnatprove`, specifying the unit and subprogram to analyze,
            # and ensuring that all kinds of proof failure yield a non-zero exit
            # code.
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
            logger.debug(spark_file_path.read_text())
            spark_data = type_checked(json.loads(spark_file_path.read_text()), dict)
        # Tally up the per-check results.
        total_proof_steps = 0
        check_results = {
            severity: Counter[str]()
            for severity in ("info", "low", "medium", "high", "warning", "error")
        }
        for check_kind in ("flow", "proof"):
            for proof_result_unchecked in type_checked(spark_data[check_kind], list):
                proof_result = type_checked(proof_result_unchecked, dict)
                check_results[proof_result["severity"]][proof_result["rule"]] += 1
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
        elif pragma_assume_count > 0:
            result = "proved_with_pragma_assume"
        else:
            result = "proved"
        # Sort counters by key for stable output
        return EvaluationStatsProve_New(
            result=result,
            proved_checks=Counter(sort_dict(proved_checks)),
            unproved_checks=Counter(sort_dict(unproved_checks)),
            warnings=Counter(sort_dict(warnings)),
            pragma_assume_count=pragma_assume_count,
            proof_steps=total_proof_steps,
        )
