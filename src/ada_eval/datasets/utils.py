import subprocess
from pathlib import Path


def git_ls_files(root: Path) -> list[Path]:
    """Get a list of files in a directory using git ls-files."""
    if not root.exists():
        return []
    result = subprocess.run(
        ["git", "ls-files", "-com", "--exclude-standard", "--deduplicate"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )

    git_files = [root / line for line in result.stdout.splitlines()]

    # We have to check that a path exists, as git ls-files will return files
    # that were previously committed but have since been deleted.
    return [path for path in git_files if path.is_file()]


def is_git_up_to_date(path: Path) -> bool:
    """
    Check if the contents of a folder are up to date in git.

    In this context we mean that no changes have been made to the files in the
    folder, including file creations/deletions/modifications.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain=1", "."],
        check=False,
        encoding="utf-8",
        capture_output=True,
        cwd=path,
    )
    return result.returncode == 0 and (
        result.stdout is None or result.stdout.strip() == ""
    )


def get_file_or_empty(path: Path) -> str:
    """Get the contents of a file, or an empty string if the file does not exist."""
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""
