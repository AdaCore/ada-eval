from ada_eval.datasets import DatasetKind


def test_dataset_type_str():
    assert str(DatasetKind.ADA) == "ada"
    assert str(DatasetKind.EXPLAIN) == "explain"
    assert str(DatasetKind.SPARK) == "spark"
