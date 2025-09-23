import logging
import shutil
import subprocess
import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

logger = logging.getLogger(__name__)


def sort_dict[K: SupportsRichComparison, V](d: dict[K, V]) -> dict[K, V]:
    """Return a copy of `d` sorted by key."""
    return dict(sorted(d.items(), key=lambda item: item[0]))


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
    Assert that `value` is an instance of `expected_type` and return it.

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
