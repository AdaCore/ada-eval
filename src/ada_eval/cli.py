import argparse
from pathlib import Path
from ada_eval.datasets.generate import generate_completions
from ada_eval.datasets.pack_unpack import pack_datasets, unpack_datasets
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR


def call_unpack_datasets(args):
    unpack_datasets(args.src, args.dest, args.force)


def call_pack_datasets(args):
    pack_datasets(args.src, args.dest)

def call_generate_completions(args):
    generate_completions(args.src)

def main():
    parser = argparse.ArgumentParser(description="CLI for Eval Framework")
    subparsers = parser.add_subparsers()

    # Unpack datasets subcommand
    unpack_parser = subparsers.add_parser("unpack", help="Unpack datasets")
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
    pack_parser = subparsers.add_parser("pack", help="Pack datasets")
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

    # Generate completions for a dataset
    generate_parser = subparsers.add_parser("generate", help="Generate completions")
    generate_parser.set_defaults(func=call_generate_completions)
    generate_parser.add_argument(
        "--src",
        type=Path,
        help="Path to dataset or dir of datasets",
        default=COMPACTED_DATASETS_DIR,
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
