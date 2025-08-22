from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BASE_DATASETS_DIR = DATA_DIR / "base"
GENERATED_DATASETS_DIR = DATA_DIR / "generated"
EVALUATED_DATASETS_DIR = DATA_DIR / "evaluated"

EXPANDED_DATASETS_DIR = BASE_DATASETS_DIR / "expanded"
COMPACTED_DATASETS_DIR = BASE_DATASETS_DIR / "compacted"

TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"
