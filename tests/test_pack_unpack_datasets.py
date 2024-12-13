import shutil
import subprocess
from pathlib import Path

from ada_eval.datasets import OTHER_JSON_NAME
from ada_eval.datasets.loader import (
    git_ls_files,
)
from ada_eval.datasets.pack_unpack import pack_datasets, unpack_datasets
from ada_eval.datasets.utils import (
    get_packed_dataset_files,
    get_unpacked_dataset_dirs,
    is_collection_of_packed_datasets,
    is_collection_of_unpacked_datasets,
    is_git_up_to_date,
    is_packed_dataset,
    is_unpacked_dataset,
    is_unpacked_sample,
)
from ada_eval.paths import COMPACTED_DATASETS_DIR, EXPANDED_DATASETS_DIR

# def setup_dataset(dataset_root: Path):
#     sample_dir = dataset_root / "sample"
#     sample_dir.mkdir(parents=True)
#     other_json_file = sample_dir / OTHER_JSON_NAME
#     other_json_file.write_text("{}", encoding="utf-8")
#     return sample_dir


# def setup_datasets(datasets_root: Path):
#     for i in ["ada", "explain", "spark"]:
#         dataset_root = datasets_root / f"{i}_test_dataset"
#         setup_dataset(dataset_root)


# def test_is_unpacked_dataset_with_valid_dataset(tmp_path: Path):
#     setup_dataset(tmp_path)
#     assert is_unpacked_dataset(tmp_path) is True


def test_is_unpacked_dataset_with_no_samples(tmp_path: Path):
    assert is_unpacked_dataset(tmp_path) is False


# def test_is_collection_of_datasets_with_valid_datasets(tmp_path: Path):
#     setup_datasets(tmp_path)
#     assert is_collection_of_unpacked_datasets(tmp_path) is True


def test_is_collection_of_datasets_with_no_datasets(tmp_path: Path):
    assert is_collection_of_unpacked_datasets(tmp_path) is False


def test_is_unpacked_sample_with_valid_sample(tmp_path: Path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    other_json_file = sample_dir / OTHER_JSON_NAME
    other_json_file.write_text("{}", encoding="utf-8")
    assert is_unpacked_sample(sample_dir) is True


def test_is_unpacked_sample_with_no_other_json(tmp_path: Path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    assert is_unpacked_sample(sample_dir) is False


def test_is_unpacked_sample_with_other_json_as_file(tmp_path: Path):
    other_json_file = tmp_path / OTHER_JSON_NAME
    other_json_file.write_text("{}", encoding="utf-8")
    assert is_unpacked_sample(other_json_file) is False


def test_is_unpacked_sample_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    assert is_unpacked_sample(non_dir_file) is False


# def test_get_unpacked_dataset_dirs_with_single_dataset(tmp_path: Path):
#     dataset_dir = tmp_path / "ada_test_dataset"
#     setup_dataset(dataset_dir)
#     datasets = get_unpacked_dataset_dirs(dataset_dir)
#     assert len(datasets) == 1
#     dataset = load_unpacked_dataset(datasets[0])
#     assert dataset.type == DatasetType.ADA


# def test_get_unpacked_dataset_dirs_with_multiple_datasets(tmp_path: Path):
#     setup_datasets(tmp_path)
#     datasets = get_unpacked_dataset_dirs(tmp_path)
#     print(datasets)
#     assert len(datasets) == 3


def test_get_unpacked_dataset_dirs_with_no_datasets(tmp_path: Path):
    datasets = get_unpacked_dataset_dirs(tmp_path)
    assert len(datasets) == 0


# def test_get_unpacked_dataset_dirs_with_invalid_dataset(tmp_path: Path):
#     dataset_dir = tmp_path / "INVALID"
#     setup_dataset(dataset_dir)
#     datasets = get_unpacked_dataset_dirs(dataset_dir)
#     assert len(datasets) == 0


def test_get_unpacked_dataset_dirs_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    datasets = get_unpacked_dataset_dirs(non_dir_file)
    assert len(datasets) == 0


# def test_get_unpacked_dataset_dirs_with_non_dataset_in_collection(tmp_path: Path):
#     setup_datasets(tmp_path)
#     invalid_dir = tmp_path / "INVALID"
#     invalid_dir.mkdir()
#     datasets = get_unpacked_dataset_dirs(tmp_path)
#     assert len(datasets) == 3


def setup_git_repo(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)


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


def create_datasets(dataset_root: Path):
    dataset_names = ["ada", "explain", "spark"]
    for name in dataset_names:
        dataset_file = dataset_root / f"{name}.jsonl"
        dataset_file.write_text("")  # TODO improve this to write a valid jsonl dataset


def test_valid_datasets(tmp_path: Path):
    create_datasets(tmp_path)
    assert is_packed_dataset(tmp_path / "ada.jsonl") is True
    assert is_collection_of_packed_datasets(tmp_path) is True


def test_invalid_datasets(tmp_path: Path):
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


def test_get_packed_datasets(tmp_path: Path):
    assert get_packed_dataset_files(tmp_path) == []
    assert get_packed_dataset_files(tmp_path / "ada.jsonl") == []
    create_datasets(tmp_path)
    assert len(get_packed_dataset_files(tmp_path)) == 3
    assert len(get_packed_dataset_files(tmp_path / "ada.jsonl")) == 1
    assert len(get_packed_dataset_files(tmp_path / "ada")) == 0


def test_pack_unpack(tmp_path: Path):
    # Packing then unpacking a datasets should result in the same dataset
    packed_dir = tmp_path / "packed"
    unpacked_dir = tmp_path / "unpacked"
    shutil.copytree(COMPACTED_DATASETS_DIR, packed_dir)
    shutil.copytree(EXPANDED_DATASETS_DIR, unpacked_dir)

    # Check that the datasets have been copied as expected
    assert is_collection_of_packed_datasets(packed_dir) is True
    assert is_collection_of_unpacked_datasets(unpacked_dir) is True

    # Create a git repo and commit the datasets
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", '"Starting state"'],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
    )

    # Small check for our is_git_up_to_date function
    assert is_git_up_to_date(tmp_path) is True

    # Remove the packed dataset
    shutil.rmtree(packed_dir)

    # Check that the packed datasets has been removed
    assert is_collection_of_packed_datasets(packed_dir) is False
    res = subprocess.run(
        ["git", "status", "--porcelain=1"],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert res.stdout.strip() != ""

    # Pack the dataset
    pack_datasets(src_dir=unpacked_dir, dest_dir=packed_dir)

    # Check that the dataset has been packed and there are no git changes
    assert is_collection_of_packed_datasets(packed_dir) is True
    res = subprocess.run(
        ["git", "status", "--porcelain=1"],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    if res.stdout.strip() != "":
        dbg = subprocess.run(
            ["git", "diff"],
            cwd=tmp_path,
            encoding="utf-8",
            capture_output=True,
        )
        print(dbg.stdout)
    assert res.stdout.strip() == ""

    # Remove the unpacked dataset
    shutil.rmtree(unpacked_dir)

    # Check that the unpacked datasets has been removed
    assert is_collection_of_unpacked_datasets(unpacked_dir) is False
    res = subprocess.run(
        ["git", "status", "--porcelain=1"],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert res.stdout.strip() != ""

    # Unpack the dataset
    unpack_datasets(src=packed_dir, dest_dir=unpacked_dir, force=True)

    # Check that the dataset has been unpacked and there are no git changes
    assert is_collection_of_unpacked_datasets(unpacked_dir) is True
    res = subprocess.run(
        ["git", "status", "--porcelain=1"],
        cwd=tmp_path,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    if res.stdout.strip() != "":
        dbg = subprocess.run(
            ["git", "diff"],
            cwd=tmp_path,
            encoding="utf-8",
            capture_output=True,
        )
        print(dbg.stdout)
    assert res.stdout.strip() == ""
