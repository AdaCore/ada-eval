import re
from pathlib import Path

import pytest

from ada_eval.utils import (
    TextPositionOutOfRangeError,
    index_from_line_and_col,
    make_files_relative_to,
)


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


def test_index_from_line_and_col():
    text = "abc\ndef\nghi"
    assert text[index_from_line_and_col(text, 1, 1)] == "a"
    assert text[index_from_line_and_col(text, 1, 2)] == "b"
    assert text[index_from_line_and_col(text, 1, 3)] == "c"
    assert text[index_from_line_and_col(text, 1, 4)] == "\n"
    assert text[index_from_line_and_col(text, 2, 2)] == "e"
    assert text[index_from_line_and_col(text, 3, 3)] == "i"

    error_msg = "column 5 is out of range [1, 4]"
    with pytest.raises(TextPositionOutOfRangeError, match=re.escape(error_msg)):
        index_from_line_and_col(text, 1, 5)
    error_msg = "line 4 is out of range [1, 3]"
    with pytest.raises(TextPositionOutOfRangeError, match=re.escape(error_msg)):
        index_from_line_and_col(text, 4, 1)
