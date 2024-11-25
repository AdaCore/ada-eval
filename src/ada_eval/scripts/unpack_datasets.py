"""Script to unpack one or more datasets from the data folder. This creates
projects structures that are easier to work with.

This is the reverse process of the pack_datasets.py script.
"""
import argparse
import json

from dataclasses import dataclass
from pathlib import Path

from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR, DATASET_TEMPLATES_DIR

@dataclass
class Args:
    pass

def pass_args() -> Args:
    pass

if __name__ == "__main__":
    pass