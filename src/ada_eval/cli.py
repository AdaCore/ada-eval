import argparse
import logging
from os import cpu_count
from pathlib import Path

from ada_eval.datasets.pack_unpack import pack_datasets, unpack_datasets
from ada_eval.paths import (
    COMPACTED_DATASETS_DIR,
    EXPANDED_DATASETS_DIR,
    GENERATED_DATASETS_DIR,
)
from ada_eval.tools import Tool, create_tool


def tool(tool_name: str) -> Tool:
    """Case-insensitive `Tool` constructor."""
    return Tool(tool_name.lower())


def call_unpack_datasets(args):
    unpack_datasets(src=args.src, dest_dir=args.dest, force=args.force)


def call_pack_datasets(args):
    pack_datasets(src_dir=args.src, dest_dir=args.dest, force=args.force)


def generate(args):
    tool = create_tool(args.tool, args.tool_config_file)
    tool.apply_to_directory(
        path=args.dataset,
        output_dir=GENERATED_DATASETS_DIR,
        jobs=args.jobs,
    )


def call_evaluate_completions(_):
    raise NotImplementedError


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="CLI for Eval Framework")
    subparsers = parser.add_subparsers(required=True)

    default_num_jobs = cpu_count() or 1

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
        type=tool,
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
    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate completions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluation_parser.set_defaults(func=call_evaluate_completions)
    evaluation_parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to packed dataset or dir of packed datasets",
        default=GENERATED_DATASETS_DIR,
    )
    evaluation_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="Number of evaluations to run in parallel.",
        default=default_num_jobs,
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
