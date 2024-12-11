from abc import ABC, abstractmethod
from pathlib import Path
from ada_eval.datasets.types.datasets import DatasetType


class GenericTool(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config_file: Path):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def supported_dataset_types(self) -> tuple[DatasetType]:
        pass

    @abstractmethod
    def apply(self, sample_working_dir: Path):
        pass
