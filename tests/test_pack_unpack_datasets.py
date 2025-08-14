import shutil
import subprocess
from pathlib import Path

from helpers import (
    assert_git_status,
    compacted_test_datasets,  # noqa: F401  # Fixtures used implicitly
    expanded_test_datasets,  # noqa: F401  # Fixtures used implicitly
    setup_git_repo,
)

from ada_eval.datasets import OTHER_JSON_NAME, SparkSample
from ada_eval.datasets.loader import load_unpacked_dataset
from ada_eval.datasets.pack_unpack import pack_datasets, unpack_datasets
from ada_eval.datasets.types.datasets import (
    get_packed_dataset_files,
    get_unpacked_dataset_dirs,
    is_collection_of_packed_datasets,
    is_collection_of_unpacked_datasets,
    is_packed_dataset,
    is_unpacked_dataset,
)
from ada_eval.datasets.types.samples import is_unpacked_sample
from ada_eval.datasets.utils import git_ls_files, is_git_up_to_date


def test_is_unpacked_dataset_with_valid_dataset(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    assert is_unpacked_dataset(expanded_test_datasets / "spark_test") is True


def test_is_unpacked_dataset_with_no_samples(tmp_path: Path):
    assert is_unpacked_dataset(tmp_path) is False


def test_is_collection_of_datasets_with_valid_datasets(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    assert is_collection_of_unpacked_datasets(expanded_test_datasets) is True


def test_is_collection_of_datasets_with_no_datasets(tmp_path: Path):
    assert is_collection_of_unpacked_datasets(tmp_path) is False


def test_is_unpacked_sample(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    sample_paths = list(expanded_test_datasets.glob("*/*"))
    assert len(sample_paths) > 0
    for path in sample_paths:
        # Test with valid samples
        assert is_unpacked_sample(path) is True
        # Test with the `other.json` file as the path
        assert is_unpacked_sample(path / OTHER_JSON_NAME) is False
        # Test without the `other.json` file
        (path / OTHER_JSON_NAME).unlink()
        assert is_unpacked_sample(path) is False


def test_is_unpacked_sample_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    assert is_unpacked_sample(non_dir_file) is False


def test_get_unpacked_dataset_dirs_with_single_dataset(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    datasets = get_unpacked_dataset_dirs(expanded_test_datasets / "spark_test")
    assert len(datasets) == 1
    dataset = load_unpacked_dataset(datasets[0])
    assert dataset.sample_type == SparkSample


def test_get_unpacked_dataset_dirs_with_multiple_datasets(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    datasets = get_unpacked_dataset_dirs(expanded_test_datasets)
    assert len(datasets) == 3


def test_get_unpacked_dataset_dirs_with_no_datasets(tmp_path: Path):
    datasets = get_unpacked_dataset_dirs(tmp_path)
    assert len(datasets) == 0


def test_get_unpacked_dataset_dirs_with_invalid_dataset(expanded_test_datasets: Path):  # noqa: F811  # pytest fixture
    dataset_dir = expanded_test_datasets / "spark_test"
    for other_json in dataset_dir.glob("**/other.json"):
        other_json.unlink()
    datasets = get_unpacked_dataset_dirs(dataset_dir)
    assert len(datasets) == 0


def test_get_unpacked_dataset_dirs_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    datasets = get_unpacked_dataset_dirs(non_dir_file)
    assert len(datasets) == 0


def test_get_unpacked_dataset_dirs_with_non_dataset_in_collection(
    expanded_test_datasets: Path,  # noqa: F811  # pytest fixture
):
    invalid_dir = expanded_test_datasets / "INVALID"
    invalid_dir.mkdir()
    datasets = get_unpacked_dataset_dirs(expanded_test_datasets)
    assert len(datasets) == 3


def test_git_ls_files_empty(tmp_path: Path):
    setup_git_repo(tmp_path)
    assert git_ls_files(tmp_path) == []


def test_git_ls_files_non_empty(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    dir_1 = tmp_path / "dir_1"
    dir_2 = tmp_path / "dir_2"
    dir_1.mkdir()
    dir_2.mkdir()
    file3 = dir_1 / "files3.txt"
    file1.write_text("content1", encoding="utf-8")
    file2.write_text("content2", encoding="utf-8")
    file3.write_text("content3", encoding="utf-8")
    assert len(git_ls_files(tmp_path)) == 3


def test_git_ls_files_staged(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    assert len(git_ls_files(tmp_path)) == 1


def test_git_ls_files_committed(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", '"foo"'], cwd=tmp_path, check=True)
    assert len(git_ls_files(tmp_path)) == 1


def test_git_ls_files_deleted(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", '"foo"'], cwd=tmp_path, check=True)
    file1.unlink()
    assert len(git_ls_files(tmp_path)) == 0


def test_git_ls_files_modified(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", '"foo"'], cwd=tmp_path, check=True)
    file1.write_text("modified content", encoding="utf-8")
    assert len(git_ls_files(tmp_path)) == 1


def test_is_packed_dataset_valid(compacted_test_datasets: Path):  # noqa: F811  # pytest fixture
    dataset_files = list(compacted_test_datasets.glob("*"))
    assert len(dataset_files) > 0
    assert all(is_packed_dataset(path) is True for path in dataset_files)
    assert is_collection_of_packed_datasets(compacted_test_datasets) is True


def test_is_packed_dataset_invalid(tmp_path: Path):
    fake_dataset = tmp_path / "ada.json"
    fake_dataset.write_text("")
    assert is_packed_dataset(fake_dataset) is False
    assert is_collection_of_packed_datasets(fake_dataset) is False
    fake_dataset = tmp_path / "ada.jsonl"
    assert is_packed_dataset(fake_dataset) is False
    assert is_collection_of_packed_datasets(fake_dataset) is False
    assert is_packed_dataset(tmp_path) is False
    assert is_collection_of_packed_datasets(tmp_path) is False
    fake_dataset.mkdir()
    assert is_packed_dataset(fake_dataset) is False
    assert is_collection_of_packed_datasets(fake_dataset) is False


def test_get_packed_dataset_files(tmp_path: Path, compacted_test_datasets: Path):  # noqa: F811  # pytest fixture
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert get_packed_dataset_files(empty_dir) == []
    assert get_packed_dataset_files(empty_dir / "ada.jsonl") == []
    assert len(get_packed_dataset_files(compacted_test_datasets)) == 3
    assert (
        len(get_packed_dataset_files(compacted_test_datasets / "ada_test.jsonl")) == 1
    )
    assert len(get_packed_dataset_files(compacted_test_datasets / "ada_test")) == 0


def test_pack_unpack(compacted_test_datasets, expanded_test_datasets, tmp_path):  # noqa: F811  # pytest fixtures
    """Packing then unpacking datasets should result in the same datasets."""
    packed_dir = compacted_test_datasets
    unpacked_dir = expanded_test_datasets

    # Check that the datasets have been copied as expected
    assert is_collection_of_packed_datasets(packed_dir) is True
    assert is_collection_of_unpacked_datasets(unpacked_dir) is True

    # Create a git repo and commit the datasets (with a small check for our
    # `is_git_up_to_date()` function)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    assert is_git_up_to_date(tmp_path) is False
    subprocess.run(
        ["git", "commit", "-m", '"Starting state"'],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
    )
    assert is_git_up_to_date(tmp_path) is True

    # Remove the packed datasets
    shutil.rmtree(packed_dir)

    # Check that the packed datasets have been removed
    assert is_collection_of_packed_datasets(packed_dir) is False
    assert_git_status(tmp_path, expect_dirty=True)

    # Pack the unpacked datasets to `packed_dir`
    pack_datasets(src_dir=unpacked_dir, dest_dir=packed_dir)

    # Check that doing so has restored the original packed dataset, and there
    # are therefore no git changes
    assert is_collection_of_packed_datasets(packed_dir) is True
    assert_git_status(tmp_path, expect_dirty=False)

    # Remove the unpacked datasets
    shutil.rmtree(unpacked_dir)

    # Check that the unpacked datasets have been removed
    assert is_collection_of_unpacked_datasets(unpacked_dir) is False
    assert_git_status(tmp_path, expect_dirty=True)

    # Unpack the packed datasets to `unpacked_dir`
    unpack_datasets(src=packed_dir, dest_dir=unpacked_dir)

    # Check that the dataset has been unpacked
    assert is_collection_of_unpacked_datasets(unpacked_dir) is True

    # There will be git changes, as we did not commit a `comments.md` or
    # `prompt.md` file for `spark_test/test_sample_2`, but these will have been
    # created (empty) by the unpacking process.
    assert_git_status(tmp_path, expect_dirty=True)
    spark_sample_2_dir = unpacked_dir / "spark_test" / "test_sample_2"
    assert (spark_sample_2_dir / "comments.md").read_text() == ""
    assert (spark_sample_2_dir / "prompt.md").read_text() == ""

    # Remove the offending files and check that the git status is clean again
    (spark_sample_2_dir / "comments.md").unlink()
    (spark_sample_2_dir / "prompt.md").unlink()
    assert_git_status(tmp_path, expect_dirty=False)
