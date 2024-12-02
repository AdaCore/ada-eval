from ada_eval.common_types import DatasetType


def test_dataset_type_str():
    assert str(DatasetType.ADA) == "ADA"
    assert str(DatasetType.EXPLAIN) == "EXPLAIN"
    assert str(DatasetType.SPARK) == "SPARK"
