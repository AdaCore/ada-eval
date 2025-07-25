from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BASE_DATASETS_DIR = DATA_DIR / "base"
GENERATED_DATASETS_DIR = DATA_DIR / "generated"
EVALUATED_DATASETS_DIR = DATA_DIR / "evaluated"
GENERATION_WORKING_DIR = DATA_DIR / "in_progress_generation"
EVALUATION_WORKING_DIR = DATA_DIR / "in_progress_evaluation"

EXPANDED_DATASETS_DIR = BASE_DATASETS_DIR / "expanded"
COMPACTED_DATASETS_DIR = BASE_DATASETS_DIR / "compacted"
