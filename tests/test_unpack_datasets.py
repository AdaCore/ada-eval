from pathlib import Path

from ada_eval.scripts.unpack_datasets import *


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


def test_get_dataset_files(tmp_path: Path):
    assert get_dataset_files(tmp_path) == []
    assert get_dataset_files(tmp_path / "ada.jsonl") == []
    create_datasets(tmp_path)
    assert len(get_dataset_files(tmp_path)) == 3
    assert len(get_dataset_files(tmp_path / "ada.jsonl")) == 1
    assert len(get_dataset_files(tmp_path / "ada")) == 0
