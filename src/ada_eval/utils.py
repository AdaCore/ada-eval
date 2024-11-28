import json
from pathlib import Path

from ada_eval.common_types import OTHER_JSON_NAME, SampleTemplate


def make_files_relative_to(path: Path, files: list[Path]) -> list[Path]:
    """Makes a list of files relative to a given path"""
    return [file.relative_to(path) for file in files]


def get_sample_template(template_dir: Path) -> SampleTemplate:
    """Returns the sample template for a dataset"""
    other_json_path = template_dir / OTHER_JSON_NAME
    other_json_contents = {}
    if other_json_path.is_file():
        other_json_contents = json.loads(other_json_path.read_text())
    files = {}
    for root, _, filenames in template_dir.walk():
        for file in filenames:
            file = root / file
            contents = file.read_text()
            file = file.relative_to(template_dir)
            files[file] = contents
    return SampleTemplate(sources=files, others=other_json_contents)
