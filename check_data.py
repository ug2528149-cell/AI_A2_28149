
from pathlib import Path

from src.validation import load_and_clean

csv_path = Path("data/AI_A2_8149.csv")

cleaned, report = load_and_clean(csv_path)

print("DATA VALIDATION SUCCESSFUL")
print("\nOriginal dataset report:")
print(report)

print("\nCleaned dataset shape:", cleaned.shape)
print("Remaining exact duplicates:", cleaned.duplicated().sum())

print("\nRemaining missing values:")
print(cleaned.isna().sum())
