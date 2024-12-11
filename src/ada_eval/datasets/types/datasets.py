from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from .samples import (
    BaseSample,
    AdaSample,
    ExplainSample,
    SparkSample,
)


# Enum that specifies the type of dataset
class DatasetType(Enum):
    ADA = "ada"
    EXPLAIN = "explain"
    SPARK = "spark"

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(kw_only=True)
class Dataset:
    name: str
    type: DatasetType
    samples: list[BaseSample]

    def save_unpacked(self, unpacked_datasets_root: Path):
        dataset_root = unpacked_datasets_root / f"{self.type}_{self.name}"
        dataset_root.mkdir(exist_ok=True, parents=True)
        for sample in self.samples:
            sample.unpack(dataset_root)

    def save_packed(self, dest_dir: Path):
        dest_file = dest_dir / f"{self.type}_{self.name}.jsonl"
        with dest_file.open("w") as f:
            for sample in self.samples:
                f.write(sample.to_json() + "\n")


class AdaDataset(Dataset):
    type = DatasetType.ADA
    samples: list[AdaSample]


class ExplainDataset(Dataset):
    type = DatasetType.EXPLAIN
    samples: list[ExplainSample]


class SparkDataset(Dataset):
    type = DatasetType.SPARK
    samples: list[SparkSample]
