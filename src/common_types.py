from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Enum that specifies the type of dataset
class DatasetType(Enum):
    ADA = "ada"
    EXPLAIN = "explain"
    SPARK = "spark"

    def __str__(self) -> str:
        return self.name

@dataclass(kw_only=True)
class Sloc:
    line: int
    column: int | None

@dataclass(kw_only=True)
class Location:
    path: Path
    start: Sloc | None
    end: Sloc | None

@dataclass(kw_only=True)
class BaseDataset:
    type: DatasetType
    name: str
    location: Location
    prompt: str
    sources: Dict[Path, str]
    canonical_solution: Any
    comments: str

@dataclass(kw_only=True)
class AdaDataset(BaseDataset):
    type: DatasetType = DatasetType.ADA
    canonical_solution: Dict[Path, str]
    test_cases: Dict[Path, str]

@dataclass(kw_only=True)
class ExplainSolution:
    reference_answer: str
    correct_statements: list[str]
    incorrect_statements: list[str]

@dataclass(kw_only=True)
class ExplainDataset(BaseDataset):
    type: DatasetType = DatasetType.EXPLAIN
    canonical_solution: ExplainSolution

@dataclass(kw_only=True)
class SparkDataset(AdaDataset):
    type: DatasetType = DatasetType.SPARK
