import subprocess
import time
from pathlib import Path


def make_files_relative_to(path: Path, files: list[Path]) -> list[Path]:
    """Make a list of files relative to a given path."""
    return [file.relative_to(path) for file in files]


def run_cmd_with_timeout(
    cmd: list[str], working_dir: Path, timeout: int
) -> tuple[subprocess.CompletedProcess[str] | None, int]:
    """
    Run a command with a timeout and return the result.

    The output will be captured with encoding set to UTF-8.

    Args:
        cmd (list[str]): The command to run.
        working_dir (Path): The directory to run the command in.
        timeout (int): The timeout in seconds.

    Returns:
        result (subprocess.CompletedProcess | None): The completed process
            object, or `None` if the process timed out.
        runtime_ms (int): The runtime of the process in milliseconds.

    """
    start = time.monotonic_ns()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            cwd=working_dir,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
        )
        end = time.monotonic_ns()
    except subprocess.TimeoutExpired:
        result = None
        end = time.monotonic_ns()
    return result, (end - start) // 1_000_000


class UnexpectedTypeError(TypeError):
    """Raised when an unexpected type is encountered."""

    def __init__(self, expected_type: type, actual_type: type):
        super().__init__(
            f"Expected type {expected_type.__name__}, but got {actual_type.__name__}."
        )


class ExecutableNotFoundError(RuntimeError):
    """Raised when a required executable is not found in the PATH."""

    def __init__(self, executable_name: str):
        super().__init__(f"'{executable_name}' is not available in the PATH.")
