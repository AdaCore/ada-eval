import logging
import re
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path

from ada_eval.datasets import (
    Dataset,
    Eval,
    EvaluatedSample,
    EvaluationStats,
    EvaluationStatsInvalid,
    EvaluationStatsProve,
    ExplainSample,
    GeneratedProofSample,
    GeneratedSample,
    ProofSample,
    Sample,
    SampleKind,
    load_datasets,
    verify_datasets_equal,
)
from ada_eval.datasets.trivial_generations import generate_canonical, generate_null
from ada_eval.datasets.types.directory_contents import DirectoryContents
from ada_eval.evals.evaluate import evaluate_datasets
from ada_eval.evals.prove import ADA_COMMENT, ada_tokens
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR
from ada_eval.utils import diff_sequences, serialise_sequence

logger = logging.getLogger(__name__)


class IncorrectDatasetError(ValueError):
    """Parent class for all dataset check exceptions."""


class FailedCanonicalEvaluationError(IncorrectDatasetError):
    """Raised when a canonical evaluation result does not represent a pass."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        failed = [
            es.eval.value for es in sample.canonical_evaluation_results if not es.passed
        ]
        super().__init__(
            f"sample '{sample.name}' in dataset '{dataset.dirname}' has "
            f"non-passing canonical evaluation results: {failed}"
        )


def check_canonical_evaluation_results(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Verify that all canonical evaluation results in `datasets` passed.

    Does not check that the results are accurate.

    Raises:
        FailedCanonicalEvaluationError: If any canonical evaluation result does
            not represent a passing result.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            for eval_stats in sample.canonical_evaluation_results:
                if not eval_stats.passed:
                    raise FailedCanonicalEvaluationError(dataset, sample)


class InaccurateCanonicalEvaluationError(IncorrectDatasetError):
    """Raised when a sample's `canonical_evaluation_results` are not accurate."""

    def __init__(
        self,
        dataset: Dataset[Sample],
        sample: Sample,
        expected: Collection[EvaluationStats],
    ) -> None:
        self.dataset, self.sample, self.expected = dataset, sample, expected
        actual_evals = {es.eval.value for es in sample.canonical_evaluation_results}
        expected_evals = {es.eval.value for es in expected}
        if actual_evals != expected_evals:
            super().__init__(
                f"sample '{sample.name}' in dataset '{dataset.dirname}' does "
                f"not have the expected set of canonical evaluation results:\n"
                f"{sorted(actual_evals)} != {sorted(expected_evals)}"
            )
        else:
            actual = sorted(sample.canonical_evaluation_results, key=lambda es: es.eval)
            expected = sorted(expected, key=lambda es: es.eval)
            diff_left, diff_right = diff_sequences(
                serialise_sequence(actual), serialise_sequence(expected)
            )
            super().__init__(
                f"mismatch found on re-evaluating sample '{sample.name}' "
                f"in dataset '{dataset.dirname}':\n\n{diff_left}\n\n{diff_right}"
            )


def check_canonical_evaluation_results_accuracy(
    datasets: Sequence[Dataset[Sample]], jobs: int
) -> None:
    """
    Verify all canonical evaluation results in `datasets` actually match the samples.

    Re-runs a full canonical evaluation to check.

    Raises:
        InaccurateCanonicalEvaluationError: If any of the current canonical
            evaluation results do not match the re-evaluated results.

    """
    generated_datasets = generate_canonical(datasets)
    reevaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in reevaluated_datasets:
        for sample in dataset.samples:
            actual = sorted(sample.canonical_evaluation_results, key=lambda es: es.eval)
            expected = sorted(sample.evaluation_results, key=lambda es: es.eval)
            if actual != expected:
                raise InaccurateCanonicalEvaluationError(dataset, sample, expected)
    # `evaluate_datasets()` only returns datasets compatible with at least one
    # eval. Incompatible datasets should have empty results.
    compatible_datasets = {d.dirname for d in reevaluated_datasets}
    for initial_dataset in datasets:
        if initial_dataset.dirname not in compatible_datasets:
            for initial_sample in initial_dataset.samples:
                if len(initial_sample.canonical_evaluation_results) > 0:
                    raise InaccurateCanonicalEvaluationError(
                        initial_dataset, initial_sample, []
                    )


class PassedBaselineEvaluationError(IncorrectDatasetError):
    """Raised when a null generation passes all evals for a sample."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        desc = (
            "the empty string for"
            if isinstance(sample, ExplainSample)
            else "the unmodified sources of"
        )
        super().__init__(
            f"all evaluations passed on {desc} sample '{sample.name}' in "
            f"dataset '{dataset.dirname}'."
        )


class InvalidBaselineEvaluationError(IncorrectDatasetError):
    """Raised when an error or timeout occurs during evaluation."""

    def __init__(self, dataset: Dataset[Sample], sample: EvaluatedSample):
        self.dataset, self.sample = dataset, sample
        bad_eval_stats = next(
            es
            for es in sample.evaluation_results
            if isinstance(es, EvaluationStatsInvalid)
        )
        super().__init__(
            f"error during baseline evaluation of sample '{sample.name}' in "
            f"dataset '{dataset.dirname}': {bad_eval_stats!r}"
        )


def check_evaluation_baseline(datasets: Sequence[Dataset[Sample]], jobs: int) -> None:
    """
    Check that a null generation fails at least one eval for all samples in `datasets`.

    Also check that these failures were not due to timeouts or errors in the
    evaluations themselves.

    Raises:
        PassedBaselineEvaluationError: If any sample passes all evals when
            evaluated with a null generation.
        InvalidBaselineEvaluationError: If any sample encounters an error or
            timeout during evaluation of the null generation.

    """
    generated_datasets = generate_null(datasets)
    evaluated_datasets = evaluate_datasets(list(Eval), generated_datasets, jobs=jobs)
    for dataset in evaluated_datasets:
        for sample in dataset.samples:
            if all(es.passed for es in sample.evaluation_results):
                raise PassedBaselineEvaluationError(dataset, sample)
            if any(
                isinstance(es, EvaluationStatsInvalid)
                for es in sample.evaluation_results
            ):
                raise InvalidBaselineEvaluationError(dataset, sample)


class UnexpectedNonSparkEntitiesError(IncorrectDatasetError):
    """The sample metadata and the prover list different non-SPARK entities."""

    def __init__(
        self, dataset: Dataset[Sample], sample: ProofSample, actual: Sequence[str]
    ):
        self.dataset, self.sample, self.actual = dataset, sample, actual
        super().__init__(
            f"sample '{sample.name}' in dataset '{dataset.dirname}' declares "
            f"'expected_non_spark_entities' "
            f"{sorted(sample.expected_non_spark_entities)}, but its canonical "
            f"solution has {sorted(actual)}. If these exclusions are intentional, "
            "record the canonical list in other.json."
        )


def check_expected_non_spark_entities(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Check the declared non-SPARK entities against the canonical proof.

    Each entry permits one entity, such as an imported stub, to fall outside
    analysis. Match the canonical list exactly, including repeated names.

    Raises:
        UnexpectedNonSparkEntitiesError: If the sample lists different entities
            from those found in the reference proof.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            if not isinstance(sample, ProofSample):
                continue
            for stats in sample.canonical_evaluation_results:
                if (
                    isinstance(stats, EvaluationStatsProve)
                    and stats.result not in ("error", "subprogram_not_found")
                    and Counter(stats.non_spark_entities)
                    != Counter(sample.expected_non_spark_entities)
                ):
                    raise UnexpectedNonSparkEntitiesError(
                        dataset, sample, stats.non_spark_entities
                    )


class ProofRuleSrcPatternError(IncorrectDatasetError):
    """Raised when a `src_pattern` is attached to a proof rule."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample, rule: str):
        self.dataset, self.sample, self.rule = dataset, sample, rule
        super().__init__(
            f"sample '{sample.name}' in dataset '{dataset.dirname}' has a "
            f"'src_pattern' on proof rule '{rule}'. Matching source text can "
            f"only pin a contract's wording, not its meaning, so a candidate "
            f"can keep the text and still weaken the property. Pin the "
            f"property with a proof harness instead."
        )


def check_src_patterns(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Verify no `required_checks` entry pins a proof rule by source text.

    A `src_pattern` can pin the text of a flow aspect such as `Depends`.
    For a proof rule, it pins contract wording rather than the property:
    the contract can be weakened without changing the matched text.

    Raises:
        ProofRuleSrcPatternError: If a `src_pattern` is attached to a rule whose
            name marks it as a proof rule.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            if not isinstance(sample, ProofSample):
                continue
            for check in sample.required_checks:
                if check.src_pattern is not None and check.rule.startswith("VC_"):
                    raise ProofRuleSrcPatternError(dataset, sample, check.rule)


class ObjectDirError(IncorrectDatasetError):
    """The sample's project does not place its objects in `obj`."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        super().__init__(
            f"the 'main.gpr' of sample '{sample.name}' in dataset "
            f"'{dataset.dirname}' lacks `for Object_Dir use \"obj\";`. The proof "
            f"evaluation reads GNATprove results and object files from 'obj'."
        )


OBJECT_DIR_TOKENS = ["for", "object_dir", "use", '"obj"', ";"]


def check_object_dirs(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Verify that each proof project builds into `obj`.

    Raises:
        ObjectDirError: If a sample's `main.gpr` sets no or another `Object_Dir`.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            if not isinstance(sample, ProofSample):
                continue
            gpr = sample.canonical_solution.files.get(Path("main.gpr"), b"")
            tokens = ada_tokens(gpr.decode("utf-8", errors="replace"))
            if not any(
                tokens[k : k + len(OBJECT_DIR_TOKENS)] == OBJECT_DIR_TOKENS
                for k in range(len(tokens))
            ):
                raise ObjectDirError(dataset, sample)


class EditableFileError(IncorrectDatasetError):
    """The reference projects disagree about which file can be edited."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample, problem: str):
        self.dataset, self.sample = dataset, sample
        super().__init__(
            f"sample '{sample.name}' in dataset '{dataset.dirname}' {problem}."
        )


class ExistingAssertionError(IncorrectDatasetError):
    """The initial editable file contains a disallowed assertion."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample, pragma: str):
        self.dataset, self.sample = dataset, sample
        super().__init__(
            f"the editable file of proof sample '{sample.name}' in dataset "
            f"'{dataset.dirname}' has a `pragma {pragma}` in its base. A solution "
            f"may remove existing assertions, so the challenge must not rely on "
            f"one; only `Loop_Variant`s may be given."
        )


class TrustedEntityInEditableFileError(IncorrectDatasetError):
    """The editable file contains a trusted declaration."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample, entity: str):
        self.dataset, self.sample = dataset, sample
        super().__init__(
            f"proof sample '{sample.name}' in dataset '{dataset.dirname}' expects "
            f"the non-SPARK entity '{entity}', which is declared in its editable "
            f"file. A solution could change what is assumed about it, so declare "
            f"it in another file."
        )


class NonTerminatingSolutionPassedError(IncorrectDatasetError):
    """An infinite loop in an editable procedure was not rejected."""

    def __init__(self, dataset: Dataset[Sample], sample: Sample):
        self.dataset, self.sample = dataset, sample
        super().__init__(
            f"an infinite loop at the end of the editable procedure of sample "
            f"'{sample.name}' in dataset '{dataset.dirname}' neither failed "
            f"proof nor changed the object code. Its proof must require "
            f"termination, for example with `Always_Terminates`, as a ghost "
            f"procedure's code is not compared."
        )


ASSERTION_PRAGMAS = frozenset(
    {"assert", "assert_and_cut", "assume", "check", "loop_invariant"}
)


def editable_unit(sample: ProofSample) -> tuple[str, str]:
    """
    Identify the editable subunit in the base sources.

    Return its kind (procedure, function or package) and qualified name in lowercase.

    Raises:
        ValueError: If the file is not a subunit.

    """
    text = sample.sources.files[Path(str(sample.editable_path))].decode("utf-8")
    tokens = ada_tokens(text)
    try:
        start = tokens.index("separate")
        parent_end = tokens.index(")", start)
        kind = tokens[parent_end + 1]
        name = tokens[parent_end + 3] if kind == "package" else tokens[parent_end + 2]
    except (ValueError, IndexError) as e:
        msg = "is not a subunit"
        raise ValueError(msg) from e
    return kind, f"{''.join(tokens[start + 2 : parent_end])}.{name}"


def check_editable_files(datasets: Sequence[Dataset[Sample]]) -> None:
    """
    Validate the editable subunit against both reference projects.

    The projects must contain the same files and agree everywhere outside the
    editable subunit. Its base may contain loop variants. Other assertions
    and declarations of trusted non-SPARK entities belong outside that file.

    Other samples that expect non-SPARK entities follow the same rules, so
    that no solution can edit the contracts of those entities.

    Raises:
        EditableFileError: If a proof sample's editable file is missing, is not
            a subunit, or is not the only file its canonical solution changes.
        ExistingAssertionError: If a base editable file has an assertion.
        TrustedEntityInEditableFileError: If an expected non-SPARK entity is
            declared in the editable file.

    """
    for dataset in datasets:
        for sample in dataset.samples:
            # Without an editable file, a solution could edit trusted contracts.
            if not isinstance(sample, ProofSample) or (
                sample.kind != SampleKind.PROOF
                and len(sample.expected_non_spark_entities) == 0
            ):
                continue
            editable = sample.editable_path
            if editable is None:
                problem = (
                    "declares no 'editable_path', which a proof sample or any "
                    "sample expecting non-SPARK entities must have"
                )
                raise EditableFileError(dataset, sample, problem)
            base, canonical = sample.sources.files, sample.canonical_solution.files
            if editable not in base or editable not in canonical:
                problem = f"lacks its editable file '{editable}' in base or solution"
                raise EditableFileError(dataset, sample, problem)
            changed = sorted(
                str(path)
                for path in base.keys() | canonical.keys()
                if path != editable and base.get(path) != canonical.get(path)
            )
            if len(changed) > 0:
                problem = f"changes files other than '{editable}': {changed}"
                raise EditableFileError(dataset, sample, problem)
            tokens = ada_tokens(base[editable].decode("utf-8"))
            for k in range(len(tokens) - 1):
                if tokens[k] == "pragma" and tokens[k + 1] in ASSERTION_PRAGMAS:
                    raise ExistingAssertionError(dataset, sample, tokens[k + 1])
            try:
                _, unit = editable_unit(sample)
            except ValueError as e:
                problem = f"has an editable file '{editable}' that {e}"
                raise EditableFileError(dataset, sample, problem) from e
            for entity in sample.expected_non_spark_entities:
                if f"{entity.casefold()}.".startswith(f"{unit}."):
                    raise TrustedEntityInEditableFileError(dataset, sample, entity)


def with_infinite_loop(sample: ProofSample) -> DirectoryContents | None:
    """
    Insert an infinite loop at the end of the canonical editable procedure.

    Return `None` for other kinds of subunit. SPARK already requires functions
    to terminate.
    """
    kind, _ = editable_unit(sample)
    if kind != "procedure":
        return None
    editable = Path(str(sample.editable_path))
    text = sample.canonical_solution.files[editable].decode("utf-8")
    # Blank the comments so that the last `end [Name];` closes the procedure.
    code = ADA_COMMENT.sub(lambda m: m.group(1) or " " * len(m.group()), text)
    end = list(re.finditer(r"\bend\b[^;]*;", code, re.IGNORECASE))[-1]
    indent = text[text.rfind("\n", 0, end.start()) + 1 : end.start()]
    indent = indent if indent.isspace() else ""
    loop = f"   loop\n{indent}      null;\n{indent}   end loop;\n{indent}"
    files = dict(sample.canonical_solution.files)
    files[editable] = (text[: end.start()] + loop + text[end.start() :]).encode()
    return DirectoryContents(files)


def check_editable_procedures_terminate(
    datasets: Sequence[Dataset[Sample]], jobs: int
) -> None:
    """
    Check that an infinite loop in each editable procedure is rejected.

    The loop must either fail proof or change the object code. Ghost
    procedures emit no executable code, so their proof must require
    termination.

    Raises:
        NonTerminatingSolutionPassedError: The added loop neither fails proof
            nor changes the object code.
        InvalidBaselineEvaluationError: The evaluator reports an error or timeout.

    """
    generated_datasets: list[Dataset[GeneratedSample]] = []
    for dataset in generate_canonical(datasets):
        samples = [
            sample.model_copy(update={"generated_solution": solution})
            for sample in dataset.samples
            if isinstance(sample, GeneratedProofSample)
            and sample.kind == SampleKind.PROOF
            and (solution := with_infinite_loop(sample)) is not None
        ]
        if len(samples) > 0:
            generated_datasets.append(
                Dataset(
                    name=dataset.name, sample_type=dataset.sample_type, samples=samples
                )
            )
    if len(generated_datasets) == 0:
        return
    logger.info("Checking that editable procedures must terminate ...")
    for dataset in evaluate_datasets([Eval.PROVE], generated_datasets, jobs=jobs):
        for sample in dataset.samples:
            for es in sample.evaluation_results:
                if isinstance(es, EvaluationStatsInvalid):
                    raise InvalidBaselineEvaluationError(dataset, sample)
                # Other failures, such as a warning, say nothing about the loop.
                if not (
                    isinstance(es, EvaluationStatsProve)
                    and (es.result == "unproved" or es.changed_object_files)
                ):
                    raise NonTerminatingSolutionPassedError(dataset, sample)


def check_base_datasets(dataset_dirs: Sequence[Path], jobs: int) -> None:
    """
    Validate the datasets and reproduce their recorded evaluation results.

    Compare the datasets across directories. Check the source patterns and
    the list of entities each sample permits outside SPARK. Rerun the
    reference solutions and verify that a null generation fails at least
    one evaluation for each sample.

    Verify that each proof project builds into `obj` and has a single
    editable subunit. Append an infinite loop to editable procedures and
    verify that the proof evaluation rejects the result.

    Raises:
        DatasetsMismatchError: The directories contain different datasets.
        ProofRuleSrcPatternError: A proof rule specifies a source pattern.
        ObjectDirError: A proof project does not build into `obj`.
        UnexpectedNonSparkEntitiesError: The declared exceptions differ from
            the non-SPARK entities in the reference proof.
        FailedCanonicalEvaluationError: A recorded reference result fails.
        InaccurateCanonicalEvaluationError: A rerun differs from the recorded result.
        PassedBaselineEvaluationError: A null generation passes every evaluation.
        InvalidBaselineEvaluationError: The evaluator reports an error or timeout.
        EditableFileError: The starter or reference violates the editable-file rule.
        ExistingAssertionError: The editable starter contains a disallowed assertion.
        TrustedEntityInEditableFileError: The editable file declares a trusted entity.
        NonTerminatingSolutionPassedError: An infinite loop is not rejected.

    """
    if len(dataset_dirs) == 0:
        logger.info("No dataset directories specified; nothing to check.")
        return
    datasets = load_datasets(dataset_dirs[0])
    # Verify that all datasets are equivalent
    dataset_names = {
        EXPANDED_DATASETS_DIR: "the expanded datasets",
        COMPACTED_DATASETS_DIR: "the compacted datasets",
    }
    for other_datasets_dir in dataset_dirs[1:]:
        verify_datasets_equal(
            datasets,
            dataset_names.get(dataset_dirs[0].resolve(), f"'{dataset_dirs[0]}'"),
            load_datasets(other_datasets_dir),
            dataset_names.get(other_datasets_dir.resolve(), f"'{other_datasets_dir}'"),
        )
    check_src_patterns(datasets)
    check_object_dirs(datasets)
    check_expected_non_spark_entities(datasets)
    check_editable_files(datasets)
    # Verify that all canonical evaluation results represent passing results
    check_canonical_evaluation_results(datasets)
    # Re-run a full canonical evaluation and verify that we get the same results
    # as those stored already.
    logger.info("Re-evaluating to check canonical evaluation results are accurate ...")
    check_canonical_evaluation_results_accuracy(datasets, jobs=jobs)
    # Check that a null generation fails at least one eval on all samples
    logger.info("Checking that null generations fail at least one eval ...")
    check_evaluation_baseline(datasets, jobs=jobs)
    # Exercise the termination obligation for each editable procedure.
    check_editable_procedures_terminate(datasets, jobs=jobs)
    # Report success
    logger.info("Base datasets are correct.")
