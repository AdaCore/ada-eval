from ada_eval.datasets import DatasetType


def test_dataset_type_str():
    assert str(DatasetType.ADA) == "ada"
    assert str(DatasetType.EXPLAIN) == "explain"
    assert str(DatasetType.SPARK) == "spark"
