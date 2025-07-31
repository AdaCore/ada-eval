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
    with pytest.raises(ValueError):
        make_files_relative_to(base_path, files)


def test_make_files_relative_to_with_empty_list():
    base_path = Path("/home/user/project")
    files: list[Path] = []
    expected: list[Path] = []
    assert make_files_relative_to(base_path, files) == expected
