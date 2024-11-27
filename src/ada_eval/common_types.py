from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from dataclass_wizard import JSONWizard

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
class Location(JSONWizard):
    path: Path
    start: Sloc | None
    end: Sloc | None

@dataclass(kw_only=True)
class SampleTemplate:
    sources: Dict[Path, str]
    others: Dict[str, Any]

@dataclass(kw_only=True)
class ExplainSolution:
    reference_answer: str
    correct_statements: list[str]
    incorrect_statements: list[str]

@dataclass(kw_only=True)
class BaseSample(JSONWizard):
    name: str
    location: Location
    prompt: str
    sources: Dict[Path, str]
    canonical_solution: Any
    comments: str

@dataclass(kw_only=True)
class AdaSample(BaseSample):
    canonical_solution: Dict[Path, str]
    unit_tests: Dict[Path, str]

@dataclass(kw_only=True)
class ExplainSample(BaseSample):
    canonical_solution: ExplainSolution

@dataclass(kw_only=True)
class SparkSample(AdaSample):
    pass
