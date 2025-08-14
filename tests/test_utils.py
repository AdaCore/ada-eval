import re
from pathlib import Path

import pytest

from ada_eval.utils import make_files_relative_to


def test_make_files_relative_to():
    base_path = Path("/home/user/project")
    files = [
        Path("/home/user/project/file1.txt"),
        Path("/home/user/project/dir/file2.txt"),
        Path("/home/user/project/dir/subdir/file3.txt"),
    ]
    expected = [
        Path("file1.txt"),
        Path("dir/file2.txt"),
        Path("dir/subdir/file3.txt"),
    ]
    assert make_files_relative_to(base_path, files) == expected


def test_make_files_relative_to_with_non_relative_files():
    base_path = Path("/home/user/project")
    files = [
        Path("/home/user/project/file1.txt"),
        Path("/home/user/other_project/file2.txt"),
    ]
    error_msg = (
        "'/home/user/other_project/file2.txt' is not in the subpath of "
        "'/home/user/project'"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        make_files_relative_to(base_path, files)


def test_make_files_relative_to_with_empty_list():
    base_path = Path("/home/user/project")
    files: list[Path] = []
    expected: list[Path] = []
    assert make_files_relative_to(base_path, files) == expected
