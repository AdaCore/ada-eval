import copy
import re
from pathlib import Path
from typing import Any

import pytest

from ada_eval.utils import (
    TextPositionOutOfRangeError,
    diff_dicts,
    index_from_line_and_col,
    make_files_relative_to,
)


def test_diff_dicts_and_sequences():
    left = {
        "a": 1,
        "b": {"c": 3, "d": [{"e": 5}, [6, 7], [8, 9], [8, 9]]},
        "f": [10, 11, 12],
        "g": [13, 14, 15],
        "h": {"i": 16, "j": 17},
        "k": [18],
        "l": "hello",
    }
    right: dict[str, Any] = copy.deepcopy(left)
    right["b"]["c"] = [30, 31]
    right["b"]["d"][0]["g"] = 50
    right["b"]["d"][2].append(9.5)
    right["b"]["d"][3].pop()
    right["f"][:2] = 11, 10
    right["k"] = ["18"]
    right["l"] = "hello!"
    right["m"] = 20
    left_diff, right_diff = diff_dicts(left, right)
    assert left_diff == {
        "b": {"c": 3, "d": [{}, [], [9]]},
        "f": [10, 11],
        "k": [18],
        "l": "hello",
    }
    assert right_diff == {
        "b": {"c": [30, 31], "d": [{"g": 50}, [9.5], []]},
        "f": [11, 10],
        "k": ["18"],
        "l": "hello!",
        "m": 20,
    }


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
