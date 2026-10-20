
"""Generate the personalized synthetic dataset for SWE 3513 Assignment 2."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

VERIFICATION_PREFIX = "AI-A2-"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

TARGET_COLUMNS = ["yield_tons", "high_risk"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS


def generate_dataset(seed: int) -> pd.DataFrame:
    """Generate the seeded dataset with required data quality problems."""
    if not isinstance(seed, int) or not 0 <= seed <= 9999:
        raise ValueError("Seed must be an integer from 0000 to 9999.")

    rng = np.random.default_rng(seed)
    n_rows = 240 + (seed % 41)

    df = pd.DataFrame(
        {
            "rainfall_mm": rng.uniform(600, 1400, n_rows),
            "avg_temp_c": rng.uniform(12, 26, n_rows),
            "soil_moisture_pct": rng.uniform(20, 80, n_rows),
            "fertilizer_kg": rng.uniform(0, 250, n_rows),
            "field_size_ha": rng.uniform(0.2, 4.0, n_rows),
            "season": rng.choice(["A", "B"], size=n_rows),
            "cooperative_support": rng.integers(0, 2, size=n_rows),
        }
    )

    season_effect = df["season"].map({"A": 0.50, "B": -0.20})
    noise = rng.normal(loc=0.0, scale=0.80, size=n_rows)

    raw_yield = (
        0.004 * df["rainfall_mm"]
        + 0.06 * df["soil_moisture_pct"]
        + 0.02 * df["fertilizer_kg"]
        + 1.30 * df["field_size_ha"]
        - 0.25 * (df["avg_temp_c"] - 18).abs()
        + 0.70 * df["cooperative_support"]
        + season_effect
        + noise
    )

    df["yield_tons"] = raw_yield.clip(lower=0.50)

    yield_per_hectare = df["yield_tons"] / df["field_size_ha"]
    median_yield_per_hectare = yield_per_hectare.median()

    df["high_risk"] = (
        yield_per_hectare < median_yield_per_hectare
    ).astype(int)

    # Missing values are introduced after the targets are generated.
    missing_count = round(0.05 * n_rows)

    soil_positions = rng.choice(
        n_rows, size=missing_count, replace=False
    )
    fertilizer_positions = rng.choice(
        n_rows, size=missing_count, replace=False
    )

    df.loc[soil_positions, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_positions, "fertilizer_kg"] = np.nan

    # Append three exact copies of the first three rows.
    df = pd.concat(
        [df, df.iloc[:3].copy()],
        ignore_index=True,
    )

    return df[REQUIRED_COLUMNS]


def save_dataset(
    df: pd.DataFrame, seed: int
) -> tuple[Path, str]:
    """Save the CSV and calculate SHA-256 from its actual bytes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / f"AI_A2_{seed:04d}.csv"
    fingerprint_path = ARTIFACTS_DIR / "data_fingerprint.txt"

    df.to_csv(
        csv_path,
        index=False,
        float_format="%.4f",
    )

    fingerprint = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()

    verification_code = f"{VERIFICATION_PREFIX}{seed:04d}"

    fingerprint_path.write_text(
        f"Verification code: {verification_code}\n"
        f"Seed: {seed}\n"
        f"Rows: {len(df)}\n"
        f"SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    return csv_path, fingerprint


def main() -> None:
    """Generate the dataset from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate the personalized SWE 3513 dataset."
    )
    parser.add_argument("--seed", type=int, default=8149)
    args = parser.parse_args()

    df = generate_dataset(args.seed)
    csv_path, fingerprint = save_dataset(df, args.seed)

    print(f"Verification code: AI-A2-{args.seed:04d}")
    print(f"Seed: {args.seed}")
    print(f"Original rows: {240 + args.seed % 41}")
    print(f"Final CSV rows: {len(df)}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
    print(
        "Missing soil moisture:",
        int(df["soil_moisture_pct"].isna().sum()),
    )
    print(
        "Missing fertilizer:",
        int(df["fertilizer_kg"].isna().sum()),
    )
    print(f"CSV: {csv_path}")
    print(f"SHA-256: {fingerprint}")


if __name__ == "__main__":
    main()
