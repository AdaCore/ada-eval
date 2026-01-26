from pathlib import Path

from helpers import setup_git_repo

from ada_eval.datasets.types.directory_contents import (
    DirectoryContents,
    get_contents,
    get_contents_git_aware,
)


def test_directorycontents():
    contents = DirectoryContents(
        {
            Path("file0"): b"File 0 content",
            Path("file1"): b"File 1 content",
            Path("subdir2/file2"): b"File 2 content",
            Path("subdir3/subsubdir3/file3"): b"File 3 content",
        }
    )
    with contents.unpacked() as tmp_dir:
        assert tmp_dir.is_dir()
        assert (tmp_dir / "file0").read_text() == "File 0 content"
        assert (tmp_dir / "file1").read_text() == "File 1 content"
        assert (tmp_dir / "subdir2" / "file2").read_text() == "File 2 content"
        assert (
            tmp_dir / "subdir3" / "subsubdir3" / "file3"
        ).read_text() == "File 3 content"
    assert not tmp_dir.exists()
    content = DirectoryContents({})
    with content.unpacked() as tmp_dir:
        assert tmp_dir.is_dir()
    assert not tmp_dir.exists()


def test_get_contents(tmp_path: Path):
    # Create a directory with some files
    (tmp_path / "file0").write_text("File 0 content")
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file1").write_text("File 1 content")
    (tmp_path / ".gitignore").write_text("gitignored_dir/\n")
    (tmp_path / "gitignored_dir").mkdir()
    (tmp_path / "gitignored_dir" / "file2").write_text("File 2 content")
    (tmp_path / "dir2" / "gitignored_dir").mkdir(parents=True)
    (tmp_path / "dir2" / "file3").write_text("File 3 content")
    (tmp_path / "dir2" / "gitignored_dir" / "file4").write_text("File 4 content")
    (tmp_path / "empty_dir").mkdir()
    # Check that `get_contents()` returns the correct `DirectoryContents`
    expected_contents = DirectoryContents(
        {
            Path("file0"): b"File 0 content",
            Path("dir1/file1"): b"File 1 content",
            Path(".gitignore"): b"gitignored_dir/\n",
            Path("gitignored_dir/file2"): b"File 2 content",
            Path("dir2/file3"): b"File 3 content",
            Path("dir2/gitignored_dir/file4"): b"File 4 content",
        }
    )
    assert get_contents(tmp_path) == expected_contents
    # Check that `get_contents_git_aware()` returns the same contents (because
    # we are not currently in a git repository)
    assert get_contents_git_aware(tmp_path) == expected_contents
    # Initialise a Git repository and check that `get_contents_git_aware()`
    # now ignores the gitignored directories
    setup_git_repo(tmp_path)
    expected_contents.files.pop(Path("gitignored_dir/file2"))
    expected_contents.files.pop(Path("dir2/gitignored_dir/file4"))
    assert get_contents_git_aware(tmp_path) == expected_contents
    # Check getting the contents of a subdirectory of the Git repo
    assert get_contents_git_aware(tmp_path / "dir1") == DirectoryContents(
        {Path("file1"): b"File 1 content"}
    )
    assert get_contents_git_aware(tmp_path / "dir2") == DirectoryContents(
        {Path("file3"): b"File 3 content"}
    )
    # Change the `.gitignore` to only ignore the root `gitignored_dir` and check
    # that this is respected
    (tmp_path / ".gitignore").write_text("/gitignored_dir/\n")
    expected_contents.files[Path(".gitignore")] = b"/gitignored_dir/\n"
    expected_contents.files[Path("dir2/gitignored_dir/file4")] = b"File 4 content"
    assert get_contents_git_aware(tmp_path) == expected_contents
    assert get_contents_git_aware(tmp_path / "dir2") == DirectoryContents(
        {
            Path("file3"): b"File 3 content",
            Path("gitignored_dir/file4"): b"File 4 content",
        }
    )
    # Check that an empty directory returns an empty `DirectoryContents`
    assert get_contents_git_aware(tmp_path / "empty_dir") == DirectoryContents({})
    assert get_contents(tmp_path / "empty_dir") == DirectoryContents({})
    # Check that a non-existent directory or a non-directory also returns empty
    assert get_contents_git_aware(tmp_path / "non_existent") == DirectoryContents({})
    assert get_contents(tmp_path / "non_existent") == DirectoryContents({})
    assert get_contents_git_aware(tmp_path / "file0") == DirectoryContents({})
    assert get_contents(tmp_path / "file0") == DirectoryContents({})
