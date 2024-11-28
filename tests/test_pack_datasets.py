from pathlib import Path

from ada_eval.scripts.pack_datasets import *


def setup_dataset(dataset_root: Path):
    sample_dir = dataset_root / "sample"
    sample_dir.mkdir(parents=True)
    other_json_file = sample_dir / OTHER_JSON_NAME
    other_json_file.write_text("{}", encoding="utf-8")
    return sample_dir


def setup_datasets(datasets_root: Path):
    for i in ["ada", "explain", "spark"]:
        dataset_root = datasets_root / f"{i}"
        setup_dataset(dataset_root)


def setup_templates(template_root: Path):
    for i in ["ada", "explain", "spark"]:
        template_dir = template_root / i
        template_dir.mkdir(parents=True)
        other_json_file = template_dir / OTHER_JSON_NAME
        other_json_file.write_text("{}", encoding="utf-8")


def test_is_dataset_with_valid_dataset(tmp_path: Path):
    setup_dataset(tmp_path)
    assert is_dataset(tmp_path) is True


def test_is_dataset_with_no_samples(tmp_path: Path):
    assert is_dataset(tmp_path) is False


def test_is_collection_of_datasets_with_valid_datasets(tmp_path: Path):
    setup_datasets(tmp_path)
    assert is_collection_of_datasets(tmp_path) is True


def test_is_collection_of_datasets_with_no_datasets(tmp_path: Path):
    assert is_collection_of_datasets(tmp_path) is False


def test_is_sample_with_valid_sample(tmp_path: Path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    other_json_file = sample_dir / OTHER_JSON_NAME
    other_json_file.write_text("{}", encoding="utf-8")
    assert is_sample(sample_dir) is True


def test_is_sample_with_no_other_json(tmp_path: Path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    assert is_sample(sample_dir) is False


def test_is_sample_with_other_json_as_file(tmp_path: Path):
    other_json_file = tmp_path / OTHER_JSON_NAME
    other_json_file.write_text("{}", encoding="utf-8")
    assert is_sample(other_json_file) is False


def test_is_sample_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    assert is_sample(non_dir_file) is False


def test_get_dataset_type_with_valid_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "ada"
    setup_dataset(dataset_dir)
    assert get_dataset_type(dataset_dir) == DatasetType.ADA


def test_get_dataset_type_with_invalid_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "INVALID"
    setup_dataset(dataset_dir)
    assert get_dataset_type(dataset_dir) is None


def test_get_dataset_type_with_no_dataset(tmp_path: Path):
    assert get_dataset_type(tmp_path) is None


def test_get_dataset_type_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    assert get_dataset_type(non_dir_file) is None


def test_get_datasets_with_single_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "ada"
    setup_dataset(dataset_dir)
    template_root = tmp_path / "templates"
    setup_templates(template_root)
    datasets = get_datasets(dataset_dir, template_root)
    assert len(datasets) == 1
    assert datasets[0].dir == dataset_dir
    assert datasets[0].type == DatasetType.ADA


def test_get_datasets_with_multiple_datasets(tmp_path: Path):
    setup_datasets(tmp_path)
    template_root = tmp_path / "templates"
    setup_templates(template_root)
    datasets = get_datasets(tmp_path, template_root)
    print(datasets)
    assert len(datasets) == 3


def test_get_datasets_with_no_datasets(tmp_path: Path):
    template_root = setup_templates(tmp_path / "templates")
    datasets = get_datasets(tmp_path, template_root)
    assert len(datasets) == 0


def test_get_datasets_with_invalid_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "INVALID"
    setup_dataset(dataset_dir)
    template_root = tmp_path / "templates"
    setup_templates(template_root)
    datasets = get_datasets(dataset_dir, template_root)
    assert len(datasets) == 0


def test_get_datasets_with_non_directory_path(tmp_path: Path):
    non_dir_file = tmp_path / "file.txt"
    non_dir_file.write_text("sample content", encoding="utf-8")
    template_root = tmp_path / "templates"
    setup_templates(template_root)
    datasets = get_datasets(non_dir_file, template_root)
    assert len(datasets) == 0


def test_get_datasets_with_non_dataset_in_collection(tmp_path: Path):
    setup_datasets(tmp_path)
    invalid_dir = tmp_path / "INVALID"
    invalid_dir.mkdir()
    template_root = tmp_path / "templates"
    setup_templates(template_root)
    datasets = get_datasets(tmp_path, template_root)
    assert len(datasets) == 3


# def test_filter_template_files_with_matching_files(tmp_path: Path):
#     dataset_dir = tmp_path / "dataset"
#     base_dir = dataset_dir / BASE_DIR_NAME
#     base_dir.mkdir(parents=True)

#     sub_dir = base_dir / "sub_dir"
#     sub_dir.mkdir()

#     file1 = base_dir / "file1.txt"
#     file1.write_text("content1", encoding="utf-8")

#     file2 = base_dir / "file2.txt"
#     file2.write_text("content2", encoding="utf-8")

#     file3 = sub_dir / "file3.txt"
#     file3.write_text("content3", encoding="utf-8")

#     sample_template = SampleTemplate(
#         sources={
#             Path("file1.txt"): "content1",
#             Path("file2.txt"): "content2",
#             Path("sub_dir/file3.txt"): "content3",
#         },
#         others={},
#     )

#     dataset = UnpackedDataSetMetadata(
#         dir=dataset_dir, type=DatasetType.ADA, sample_template=sample_template
#     )

#     full_paths = [file1, file2, file3]
#     short_paths = make_files_relative_to(base_dir, full_paths)
#     result = filter_template_files(zip(short_paths, full_paths), dataset)
#     assert result == []


# def test_filter_template_files_with_different_file_names(tmp_path: Path):
#     dataset_dir = tmp_path / "dataset"
#     base_dir = dataset_dir / BASE_DIR_NAME
#     base_dir.mkdir(parents=True)

#     sub_dir = base_dir / "sub_dir"
#     sub_dir.mkdir()

#     file1 = base_dir / "file1.txt"
#     file1.write_text("content1", encoding="utf-8")

#     file2 = base_dir / "file2.txt"
#     file2.write_text("content2", encoding="utf-8")

#     file3 = sub_dir / "file3.txt"
#     file3.write_text("content3", encoding="utf-8")

#     sample_template = SampleTemplate(
#         sources={
#             Path("file1.txt"): "content1",
#             Path("file2.txt"): "content2",
#             Path("sub_dir/file4.txt"): "content3",
#         },
#         others={},
#     )

#     dataset = UnpackedDataSetMetadata(
#         dir=dataset_dir, type=DatasetType.ADA, sample_template=sample_template
#     )

#     full_paths = [file1, file2, file3]
#     short_paths = make_files_relative_to(base_dir, full_paths)
#     result = filter_template_files(zip(short_paths, full_paths), dataset)
#     assert result == [file3]


# def test_filter_template_files_with_different_contents(tmp_path: Path):
#     dataset_dir = tmp_path / "dataset"
#     base_dir = dataset_dir / BASE_DIR_NAME
#     base_dir.mkdir(parents=True)

#     sub_dir = base_dir / "sub_dir"
#     sub_dir.mkdir()

#     file1 = base_dir / "file1.txt"
#     file1.write_text("content1", encoding="utf-8")

#     file2 = base_dir / "file2.txt"
#     file2.write_text("content2", encoding="utf-8")

#     file3 = sub_dir / "file3.txt"
#     file3.write_text("content3", encoding="utf-8")

#     sample_template = SampleTemplate(
#         sources={
#             Path("file1.txt"): "content1",
#             Path("file2.txt"): "content2",
#             Path("sub_dir/file3.txt"): "different content",
#         },
#         others={},
#     )

#     dataset = UnpackedDataSetMetadata(
#         dir=dataset_dir, type=DatasetType.ADA, sample_template=sample_template
#     )

#     full_paths = [file1, file2, file3]
#     short_paths = make_files_relative_to(base_dir, full_paths)
#     result = filter_template_files(zip(short_paths, full_paths), dataset)
#     assert result == [file3]


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
    subprocess.run(["git", "commit", '-m "foo"'], cwd=tmp_path, check=True)
    assert len(git_ls_files(tmp_path)) == 1


def test_git_ls_files_deleted(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", '-m "foo"'], cwd=tmp_path, check=True)
    file1.unlink()
    assert len(git_ls_files(tmp_path)) == 0


def test_git_ls_files_modified(tmp_path: Path):
    setup_git_repo(tmp_path)
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1", encoding="utf-8")
    subprocess.run(["git", "add", str(file1)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", '-m "foo"'], cwd=tmp_path, check=True)
    file1.write_text("modified content", encoding="utf-8")
    assert len(git_ls_files(tmp_path)) == 1
