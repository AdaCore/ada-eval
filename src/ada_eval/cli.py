import argparse
from pathlib import Path
from ada_eval.unpack_datasets import unpack_datasets as unpack
from ada_eval.pack_datasets import pack_datasets as pack
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR


def unpack_datasets(args):
    unpack(args.src_dir, args.dest_dir, args.force)


def pack_datasets(args):
    pack(args.src_dir, args.dest_dir)


def main():
    parser = argparse.ArgumentParser(description="CLI for Eval Framework")
    subparsers = parser.add_subparsers()

    # Unpack datasets subcommand
    unpack_parser = subparsers.add_parser("unpack", help="Unpack datasets")
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
    unpack_parser.set_defaults(func=unpack_datasets)

    # Pack datasets subcommand
    pack_parser = subparsers.add_parser("pack", help="Pack datasets")
    pack_parser.set_defaults(func=pack_datasets)
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

    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
