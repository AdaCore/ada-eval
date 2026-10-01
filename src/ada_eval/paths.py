from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BASE_DATASETS_DIR = DATA_DIR / "base"
GENERATED_DATASETS_DIR = DATA_DIR / "generated"
EVALUATED_DATASETS_DIR = DATA_DIR / "evaluated"
SMOKE_TEST_DIR = DATA_DIR / "smoke-test"

# The expanded datasets are the source of truth
EXPANDED_DATASETS_DIR = BASE_DATASETS_DIR / "expanded"
COMPACTED_DATASETS_DIR = BASE_DATASETS_DIR / "compacted"

TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"
TOOL_CONFIGS_DIR = PROJECT_ROOT / "tools" / "configs"
