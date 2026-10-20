
"""Leakage-safe preprocessing for harvest prediction models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]

CATEGORICAL_FEATURES = ["season"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_FEATURES = {
    "yield_tons",
    "yield_per_hectare",
    "high_risk",
}


def validate_feature_names(feature_names: list[str]) -> None:
    """Reject target leakage and unknown input feature names."""
    leaked = FORBIDDEN_FEATURES.intersection(feature_names)

    if leaked:
        raise ValueError(
            f"Target leakage detected: {sorted(leaked)}"
        )

    unknown = set(feature_names) - set(MODEL_FEATURES)

    if unknown:
        raise ValueError(
            f"Unknown model features: {sorted(unknown)}"
        )

    missing = set(MODEL_FEATURES) - set(feature_names)

    if missing:
        raise ValueError(
            f"Missing required model features: {sorted(missing)}"
        )


def make_preprocessor() -> ColumnTransformer:
    """Create numeric and categorical preprocessing pipelines.

    Imputers, scalers and encoders are fitted only when the returned
    transformer is fitted on the training partition.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(estimator) -> Pipeline:
    """Combine preprocessing and a model into one reusable pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )
