
"""FastAPI service for Musanze Harvest Intelligence."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VERIFICATION_CODE = "AI-A2-8149"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

regression_path = MODELS_DIR / "regression_model.joblib"
classification_path = MODELS_DIR / "classification_model.joblib"
clustering_path = MODELS_DIR / "clustering_model.joblib"

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        "A classroom decision-support prototype using synthetic "
        "potato-farm data. Predictions require human review."
    ),
    version="1.0.0",
)


class FarmInput(BaseModel):
    """Validated farm characteristics for prediction."""

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(gt=0.2, le=4.0)
    season: str
    cooperative_support: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_categories(self):
        if self.season not in ("A", "B"):
            raise ValueError("season must be 'A' or 'B'.")

        if self.cooperative_support not in (0, 1):
            raise ValueError("cooperative_support must be 0 or 1.")

        return self


def load_models():
    """Load the trained models, failing clearly if files are missing."""
    paths = [regression_path, classification_path, clustering_path]
    missing = [str(path) for path in paths if not path.is_file()]

    if missing:
        raise RuntimeError(
            "Trained model files are missing. Run run_all.py first: "
            + ", ".join(missing)
        )

    regression = joblib.load(regression_path)
    classification = joblib.load(classification_path)
    clustering_bundle = joblib.load(clustering_path)

    return regression, classification, clustering_bundle


def predict_farm(data: FarmInput) -> dict:
    """Return yield, risk, risk probability, cluster and identity."""
    try:
        regression, classification, cluster_bundle = load_models()

        frame = pd.DataFrame([data.model_dump()], columns=FEATURES)

        yield_estimate = float(regression.predict(frame)[0])
        risk_prediction = int(classification.predict(frame)[0])

        probabilities = classification.predict_proba(frame)[0]
        classes = list(classification.classes_)
        risk_probability = float(probabilities[classes.index(1)])

        cluster_preprocessor = cluster_bundle["preprocessor"]
        cluster_model = cluster_bundle["model"]

        transformed = cluster_preprocessor.transform(frame)
        cluster = int(cluster_model.predict(transformed)[0])

        return {
            "verification_code": VERIFICATION_CODE,
            "predicted_yield_tons": round(yield_estimate, 4),
            "high_risk_prediction": risk_prediction,
            "high_risk_probability": round(risk_probability, 4),
            "cluster": cluster,
            "notice": (
                "Synthetic-data prototype only; obtain qualified "
                "human review before making farm decisions."
            ),
        }

    except (OSError, ValueError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction could not be completed: {exc}",
        ) from exc


@app.get("/health")
def health():
    """Report service status."""
    return {
        "status": "ok",
        "verification_code": VERIFICATION_CODE,
    }


@app.get("/model-info")
def model_info():
    """Return model identity and intended use."""
    try:
        regression, classification, cluster_bundle = load_models()

        return {
            "verification_code": VERIFICATION_CODE,
            "regression_model": type(regression.named_steps["model"]).__name__,
            "classification_model": type(
                classification.named_steps["model"]
            ).__name__,
            "cluster_count": int(cluster_bundle["model"].n_clusters),
            "features": FEATURES,
            "dataset_type": "synthetic",
            "intended_use": "Educational decision-support prototype",
        }

    except (OSError, RuntimeError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post("/predict")
def predict(data: FarmInput):
    """Validate a farm record and return model predictions."""
    return predict_farm(data)
