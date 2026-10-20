
"""Schema validation and cleaning for the harvest dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
}

NUMERIC_RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
}

ALLOWED_MISSING_COLUMNS = {
    "soil_moisture_pct",
    "fertilizer_kg",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Validate required columns, allowed missingness and data ranges."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    extra_columns = set(df.columns) - REQUIRED_COLUMNS
    if extra_columns:
        raise ValueError(
            f"Unexpected columns found: {sorted(extra_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    # Only the two designated feature columns may contain missing values.
    for column in df.columns:
        if column not in ALLOWED_MISSING_COLUMNS and df[column].isna().any():
            raise ValueError(
                f"Unexpected missing values in column '{column}'."
            )

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        observed = values.dropna()

        if len(observed) != df[column].notna().sum():
            raise ValueError(
                f"Column '{column}' contains non-numeric values."
            )

        if not observed.between(minimum, maximum).all():
            raise ValueError(
                f"Column '{column}' must be between "
                f"{minimum} and {maximum}."
            )

    if not df["season"].isin(["A", "B"]).all():
        raise ValueError("Season must contain only A or B.")

    if not df["cooperative_support"].isin([0, 1]).all():
        raise ValueError("cooperative_support must contain only 0 or 1.")

    if not df["high_risk"].isin([0, 1]).all():
        raise ValueError("high_risk must contain only 0 or 1.")

    if not pd.to_numeric(
        df["yield_tons"], errors="coerce"
    ).notna().all():
        raise ValueError("yield_tons must contain numeric values.")

    if (df["yield_tons"] < 0.50).any():
        raise ValueError("yield_tons cannot be below 0.50.")


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return data-quality statistics without modifying the dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "column_names": list(df.columns),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates while preserving missing values for imputation."""
    validate_schema(df)

    cleaned = df.drop_duplicates().reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No records remain after duplicate removal.")

    return cleaned


def load_and_clean(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load, validate, inspect and clean a CSV dataset."""
    df = pd.read_csv(csv_path)

    report = inspect_dataset(df)
    validate_schema(df)

    cleaned = clean_dataset(df)

    report["rows_after_cleaning"] = int(len(cleaned))
    report["duplicates_removed"] = int(len(df) - len(cleaned))

    return cleaned, report
