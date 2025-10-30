import logging
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

logger = logging.getLogger(__name__)


AnyList = list[object]
AnyDict = dict[object, object]


def sort_dict[K: SupportsRichComparison, V](d: dict[K, V]) -> dict[K, V]:
    """Return a copy of `d` sorted by key."""
    return dict(sorted(d.items(), key=lambda item: item[0]))


def diff_dicts[K, V](
    dict1: dict[K, V], dict2: dict[K, V]
) -> tuple[dict[K, V | AnyDict | AnyList], dict[K, V | AnyDict | AnyList]]:
    """
    Return the differences between two dictionaries.

    Returns the remainders of the two dicts after removing the keys which are
    present in both and have the same value.

    Nested `dict`s and `Sequence`s (except strings) are diffed recursively.
    """
    diff1: dict[K, V | AnyDict | AnyList] = {
        k: v for k, v in dict1.items() if k not in dict2 or dict2[k] != v
    }
    diff2: dict[K, V | AnyDict | AnyList] = {
        k: v for k, v in dict2.items() if k not in dict1 or dict1[k] != v
    }
    for k in diff1.keys() & diff2.keys():
        v1, v2 = diff1[k], diff2[k]
        if type(v1) is type(v2) and not isinstance(v1, str):
            if isinstance(v1, dict) and isinstance(v2, dict):
                diff1[k], diff2[k] = diff_dicts(v1, v2)
            elif isinstance(v1, Sequence) and isinstance(v2, Sequence):
                diff1[k], diff2[k] = diff_sequences(v1, v2)
    return diff1, diff2


def diff_sequences[T](
    seq1: Sequence[T], seq2: Sequence[T]
) -> tuple[list[T | AnyDict | AnyList], list[T | AnyDict | AnyList]]:
    """
    Return the differences between two sequences.

    Returns the remainders of the two sequences after removing the elements
    which are the same in both.

    Nested `dict`s and `Sequence`s (except strings) are diffed recursively.
    """
    diff1: list[T | AnyDict | AnyList] = []
    diff2: list[T | AnyDict | AnyList] = []
    for elem1, elem2 in zip(seq1, seq2, strict=False):
        if elem1 != elem2:
            new_elem1: T | AnyDict | AnyList = elem1
            new_elem2: T | AnyDict | AnyList = elem2
            if type(elem1) is type(elem2) and not isinstance(elem1, str):
                if isinstance(elem1, dict) and isinstance(elem2, dict):
                    new_elem1, new_elem2 = diff_dicts(elem1, elem2)
                elif isinstance(elem1, Sequence) and isinstance(elem2, Sequence):
                    new_elem1, new_elem2 = diff_sequences(elem1, elem2)
            diff1.append(new_elem1)
            diff2.append(new_elem2)
    if len(seq1) > len(seq2):
        diff1.extend(seq1[len(seq2) :])
    elif len(seq2) > len(seq1):
        diff2.extend(seq2[len(seq1) :])
    return diff1, diff2


def subtract_counters[T](minuend: Counter[T], subtrahend: Counter[T]) -> Counter[T]:
    """
    Equivalent to `minuend - subtrahend`, except negative counts are not dropped.

    Zero counts are still dropped.
    """
    difference = Counter[T]()
    for key in minuend.keys() | subtrahend.keys():
        count = minuend[key] - subtrahend[key]
        if count != 0:
            difference[key] = count
    return difference


def make_files_relative_to(path: Path, files: list[Path]) -> list[Path]:
    """Make a list of files relative to a given path."""
    return [file.relative_to(path) for file in files]


class TextPositionOutOfRangeError(ValueError):
    """Raised when a line or column number is out of range."""

    def __init__(self, kind: Literal["line", "column"], value: int, upper_limit: int):
        super().__init__(f"{kind} {value} is out of range [1, {upper_limit}].")


def index_from_line_and_col(text: str, line: int, col: int) -> int:
    """Return the index of the character at the given line and column in the text."""
    lines = text.splitlines(keepends=True)
    if not (1 <= line <= len(lines)):
        raise TextPositionOutOfRangeError("line", line, len(lines))
    line_str = lines[line - 1]
    if not (1 <= col <= len(line_str)):
        raise TextPositionOutOfRangeError("column", col, len(line_str))
    return sum(len(line) for line in lines[: line - 1]) + col - 1


def serialise_sequence(seq: Sequence[BaseModel]) -> list[dict[str, object]]:
    """Serialise a sequence of Pydantic models to a list of dictionaries."""
    return [item.model_dump() for item in seq]


def construct_enum_case_insensitive(cls: type[Enum], value: object) -> Enum | None:
    """Construct a member of `cls` from `value` case-insensitively."""
    if not isinstance(value, str):
        return None
    return next(
        (
            member
            for member in cls
            if isinstance(member.value, str) and member.value.lower() == value.lower()
        ),
        None,
    )


def run_cmd_with_timeout(
    cmd: list[str],
    working_dir: Path,
    timeout: int,
    *,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], int]:
    """
    Run a command with a timeout and return the result.

    The output will be captured with encoding set to UTF-8.

    Args:
        cmd (list[str]): The command to run.
        working_dir (Path): The directory to run the command in.
        timeout (int): The timeout in seconds.
        check (bool): Whether to raise a `subprocess.CalledProcessError` if the
            command returns a non-zero exit code.
        env (dict[str, str] | None): environment variables to set for the process

    Returns:
        result (subprocess.CompletedProcess): The completed process object.
        runtime_ms (int): The runtime of the process in milliseconds.

    Raises:
        subprocess.TimeoutExpired: If the timeout is exceeded.
        subprocess.CalledProcessError: If `check` is `True` and the command
            returns a non-zero exit code.

    """
    logger.debug("Running command: %s", cmd)
    start = time.monotonic_ns()
    result = subprocess.run(
        cmd,
        check=check,
        cwd=working_dir,
        capture_output=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
    )
    end = time.monotonic_ns()
    logger.debug("Return code: %d", result.returncode)
    if result.stdout:
        logger.debug("Stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("Stderr:\n%s", result.stderr)
    return result, (end - start) // 1_000_000


class UnexpectedTypeError(TypeError):
    """Raised when an unexpected type is encountered."""

    def __init__(self, expected_type: type, actual_type: type):
        super().__init__(
            f"Expected type {expected_type.__name__}, but got {actual_type.__name__}."
        )


def type_checked[T](value: object, expected_type: type[T]) -> T:
    """
    Verify that `value` is an instance of `expected_type` and return it.

    Raises:
        UnexpectedTypeError: If `value` is not of type `expected_type`.

    """
    if isinstance(value, expected_type):
        return value
    raise UnexpectedTypeError(expected_type, type(value))


class ExecutableNotFoundError(RuntimeError):
    """Raised when a required executable is not found in the PATH."""

    def __init__(self, executable_name: str):
        super().__init__(f"'{executable_name}' is not available in the PATH.")


def check_on_path(executable_name: str) -> None:
    """
    Check that an executable is available in the PATH.

    Args:
        executable_name (str): The name of the executable to check.

    Raises:
        ExecutableNotFoundError: If the executable is not found in the PATH.

    """
    if shutil.which(executable_name) is None:
        raise ExecutableNotFoundError(executable_name)
