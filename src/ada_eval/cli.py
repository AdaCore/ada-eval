import argparse
import logging
from os import cpu_count
from pathlib import Path

from ada_eval.check_datasets import check_base_datasets
from ada_eval.datasets.pack_unpack import pack_datasets, unpack_datasets
from ada_eval.datasets.types import Eval, SampleKind
from ada_eval.evals import evaluate_directory
from ada_eval.paths import (
    COMPACTED_DATASETS_DIR,
    EVALUATED_DATASETS_DIR,
    EXPANDED_DATASETS_DIR,
    GENERATED_DATASETS_DIR,
)
from ada_eval.report import report_evaluation_results
from ada_eval.tools import Tool, create_tool


def call_unpack_datasets(args):
    unpack_datasets(src=args.src, dest_dir=args.dest, force=args.force)


def call_pack_datasets(args):
    pack_datasets(src_dir=args.src, dest_dir=args.dest, force=args.force)


def generate(args):
    tool = create_tool(args.tool, args.tool_config_file)
    tool.apply_to_directory(
        path=args.dataset, output_dir=GENERATED_DATASETS_DIR, jobs=args.jobs
    )


def evaluate(args):
    if args.evals is None:
        args.evals = list(Eval)
    if args.dataset is None:
        args.dataset = (
            EXPANDED_DATASETS_DIR if args.canonical else GENERATED_DATASETS_DIR
        )
    evaluate_directory(
        evals=args.evals,
        path=args.dataset,
        output_dir=args.dataset if args.canonical else EVALUATED_DATASETS_DIR,
        jobs=args.jobs,
        canonical_evaluation=args.canonical,
    )


def call_check_base_datasets(args) -> None:
    check_base_datasets(dataset_dirs=args.datasets, jobs=args.jobs)


def call_report_evaluation_results(args) -> None:
    report_evaluation_results(
        dataset_dirs=args.dataset_dirs,
        datasets_filter=None if args.datasets is None else set(args.datasets),
        dataset_kinds_filter=(
            None if args.dataset_kinds is None else set(args.dataset_kinds)
        ),
        samples_filter=None if args.samples is None else set(args.samples),
        metrics_filter=args.with_metric,
        list_samples=args.list_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for Eval Framework")
    subparsers = parser.add_subparsers(required=True)

    default_num_jobs = cpu_count() or 1

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose logging "
            "(--jobs=1 is recommended to avoid interleaved output)"
        ),
    )

    # Unpack datasets subcommand
    unpack_parser = subparsers.add_parser(
        "unpack",
        help="Unpack datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    unpack_parser.set_defaults(func=call_unpack_datasets)
    unpack_parser.add_argument(
        "--src",
        type=Path,
        help="Path to packed dataset or dir of packed datasets",
        default=COMPACTED_DATASETS_DIR,
    )
    unpack_parser.add_argument(
        "--dest",
        type=Path,
        help="Destination dir for unpacked datasets",
        default=EXPANDED_DATASETS_DIR,
    )
    unpack_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force unpacking even if there are uncommitted changes",
    )

    # Pack datasets subcommand
    pack_parser = subparsers.add_parser(
        "pack",
        help="Pack datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pack_parser.set_defaults(func=call_pack_datasets)
    pack_parser.add_argument(
        "--src",
        type=Path,
        help="Source dir containing unpacked dataset or datasets",
        default=EXPANDED_DATASETS_DIR,
    )
    pack_parser.add_argument(
        "--dest",
        type=Path,
        help="Destination dir for packed dataset or datasets",
        default=COMPACTED_DATASETS_DIR,
    )
    pack_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force packing even if there are uncommitted changes",
    )

    # Generate completions for a dataset
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate completions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate_parser.set_defaults(func=generate)
    generate_parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Path to packed dataset or dir of packed datasets "
            "to generate completions for"
        ),
        default=COMPACTED_DATASETS_DIR,
    )
    generate_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="Number of samples to generate completions for in parallel",
        default=default_num_jobs,
    )
    generate_parser.add_argument(
        "--tool",
        type=Tool,
        choices=list(Tool),
        help="Name of tool to use for generation",
        required=True,
    )
    generate_parser.add_argument(
        "--tool-config-file",
        type=Path,
        help="Path to tool configuration file",
        required=True,
    )

    # Evaluate completions for a dataset
    evaluation_parser = subparsers.add_parser("evaluate", help="Evaluate completions")
    evaluation_parser.set_defaults(func=evaluate)
    evaluation_parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Evaluate the canonical solution instead of the generated samples. "
            "The results will be recorded in the 'canonical_evaluation_results' "
            "of each sample in the original dataset files. If there are results "
            "from the same eval(s) already present, they will be overwritten."
        ),
    )
    evaluation_parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Path to a dataset or a directory of datasets. (Default: "
            f"'{GENERATED_DATASETS_DIR}', or '{EXPANDED_DATASETS_DIR}' if "
            "'--canonical' is set)"
        ),
        default=None,
    )
    evaluation_parser.add_argument(
        "--evals",
        type=Eval,
        choices=list(Eval),
        nargs="*",
        help="Names of the evals to run. (Default: all)",
    )
    evaluation_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help=f"Number of evaluations to run in parallel. (Default: {default_num_jobs})",
        default=default_num_jobs,
    )

    # Check correctness of base datasets
    check_datasets_parser = subparsers.add_parser(
        "check-datasets",
        help="Check that base datasets are correct and equivalent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    check_datasets_parser.set_defaults(func=call_check_base_datasets)
    check_datasets_parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        help="Paths of dataset directories to check.",
        default=[EXPANDED_DATASETS_DIR, COMPACTED_DATASETS_DIR],
    )
    check_datasets_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="Number of evaluations to run in parallel.",
        default=default_num_jobs,
    )

    # Report evaluation results
    report_parser = subparsers.add_parser(
        "report", help="Generate a report of evaluation results"
    )
    report_parser.set_defaults(func=call_report_evaluation_results)
    report_parser.add_argument(
        "--dataset-dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="Paths to dataset directories to include in the report.",
        default=[EVALUATED_DATASETS_DIR],
    )
    report_parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        metavar="DATASET",
        help="Full names (i.e. '<kind>_<name>') of datasets to include in the report.",
    )
    report_parser.add_argument(
        "--dataset-kinds",
        type=SampleKind,
        nargs="+",
        metavar="KIND",
        help="Kinds of datasets to include in the report.",
    )
    report_parser.add_argument(
        "--samples",
        type=str,
        nargs="+",
        metavar="SAMPLE",
        help="Names of samples to include in the report.",
    )
    report_parser.add_argument(
        "--with-metric",
        type=str,
        action="append",
        nargs="+",
        metavar="METRIC",
        help=(
            "Include only samples for which the specified metric is present. "
            "To specify a nested metric, provide the path to the metric as the "
            "argument list. Can be specified multiple times to select the "
            'intersection. e.g. \'--with-metric build compiled "no warnings" '
            "--with-metric prove' will select all samples that compiled with no "
            "warnings and for which a 'prove' eval result is present."
        ),
    )
    report_parser.add_argument(
        "--list-samples",
        action="store_true",
        help=(
            "Instead of displaying metric values, simply list the names of the "
            "samples that match the filters."
        ),
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
