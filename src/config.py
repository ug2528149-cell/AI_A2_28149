from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

STUDENT_ID = "ug2424671@ines.ac.rw"
SEED = 8149
VERIFICATION_CODE = "AI-A2-8149"

DATA_FILE = DATA_DIR / "AI_A2_8149.csv"

RANDOM_STATE = SEED
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]

CATEGORICAL_FEATURES = [
    "season",
    "cooperative_support",
]

REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"

TARGET_COLUMNS = [
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
]

REQUIRED_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + TARGET_COLUMNS
)

for directory in (
    MODELS_DIR,
    ARTIFACTS_DIR,
    PLOTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)