
from pathlib import Path

from src.preprocessing import (
    MODEL_FEATURES,
    make_preprocessor,
    validate_feature_names,
)
from src.validation import load_and_clean


csv_path = Path("data/AI_A2_8149.csv")
df, report = load_and_clean(csv_path)

X = df[MODEL_FEATURES]

# The preprocessor is fitted here only for this smoke test.
preprocessor = make_preprocessor()
transformed = preprocessor.fit_transform(X)

print("PREPROCESSING TEST SUCCESSFUL")
print("Rows:", X.shape[0])
print("Original feature count:", X.shape[1])
print("Transformed feature shape:", transformed.shape)
print("Contains missing values:", bool(__import__("numpy").isnan(transformed).any()))

try:
    validate_feature_names(
        MODEL_FEATURES + ["yield_tons"]
    )
except ValueError as exc:
    print("Leakage safeguard successful:", exc)
else:
    raise AssertionError("Target leakage was not detected.")
