from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from pydantic import PlainSerializer, RootModel
from pydantic.functional_validators import BeforeValidator

from ada_eval.datasets.utils import git_ls_files, is_in_git_worktree


def decode_bytes(v: Any) -> Any:
    """
    Decode base64 strings to bytes, pass through bytes as-is.

    Raises:
        binascii.Error: If a string is not valid base64

    """

    if isinstance(v, str):
        return base64.b64decode(v, validate=True)
    if isinstance(v, bytes):
        return v
    return v


def encode_bytes(v: bytes) -> str:
    """Encode bytes to base64 string for JSON serialization."""
    return base64.b64encode(v).decode("utf-8")


# Bytes type that serializes to/from base64 strings for storing in JSON objects
Base64Bytes = Annotated[
    bytes,
    BeforeValidator(decode_bytes),
    PlainSerializer(encode_bytes, return_type=str, when_used="json"),
]


class _UnpackedDirectoryContextManager:
    """
    Context manager for unpacking a `DirectoryContents` to a temp directory.

    Returns the `Path` to the temp directory on entry, and cleans it up on exit.
    """

    contents: DirectoryContents
    temp_dir: TemporaryDirectory[str] | None = None

    def __init__(self, contents: DirectoryContents):
        self.contents = contents
        self.temp_dir = None

    def __enter__(self) -> Path:
        self.temp_dir = TemporaryDirectory()
        temp_dir_path = Path(self.temp_dir.__enter__())
        temp_dir_path = temp_dir_path.resolve()  # Don't return symlinks on macOS
        self.contents.unpack_to(temp_dir_path)
        return temp_dir_path

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_dir is not None:
            self.temp_dir.__exit__(exc_type, exc_value, traceback)
            self.temp_dir = None


class DirectoryContents(RootModel[dict[Path, Base64Bytes]]):
    """
    The contents of a directory.

    Attributes:
        root (dict[Path, Base64Bytes]): A mapping of the files' relative paths to their
            contents.
        files (dict[Path, Base64Bytes]): More descriptive alias for `root`.

    """

    root: dict[Path, Base64Bytes]

    @property
    def files(self) -> dict[Path, Base64Bytes]:
        return self.root

    def unpack_to(self, dest_dir: Path):
        """Unpack the contents into the specified directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)  # Should exist even if empty
        for rel_path, contents in self.files.items():
            full_path = dest_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with full_path.open("wb") as f:
                f.write(contents)

    def unpacked(self) -> _UnpackedDirectoryContextManager:
        """Return a context manager that unpacks the contents to a temp directory."""
        return _UnpackedDirectoryContextManager(self)


def get_contents(root: Path) -> DirectoryContents:
    """Return the contents of a directory."""
    if not root.is_dir():
        return DirectoryContents({})
    full_paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    files = {p.relative_to(root): p.read_bytes() for p in full_paths}
    return DirectoryContents(files)


def get_contents_git_aware(root: Path) -> DirectoryContents:
    """
    Return the contents of a directory.

    If `root` is inside a Git worktree, excludes any files that are ignored by
    Git.
    """
    if not root.is_dir():
        return DirectoryContents({})
    if not is_in_git_worktree(root):
        return get_contents(root)
    full_paths = sorted(git_ls_files(root))
    files = {p.relative_to(root): p.read_bytes() for p in full_paths}
    return DirectoryContents(files)
