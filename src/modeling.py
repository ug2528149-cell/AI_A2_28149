
"""Train, evaluate and save the harvest prediction models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    MODEL_FEATURES,
    make_model_pipeline,
    make_preprocessor,
    validate_feature_names,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

RISK_TARGET = "high_risk"
YIELD_TARGET = "yield_tons"


def _write_json(path: Path, data: dict) -> None:
    """Write a JSON artifact with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _prepare_directories() -> None:
    """Create directories used for model and evaluation artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(
    df: pd.DataFrame,
    seed: int,
) -> dict:
    """Train regression, classification and clustering models.

    The supervised tasks use the same fixed train/test row split.
    Every supervised preprocessing pipeline is fitted on training
    data only. Clustering is performed independently without targets.
    """
    _prepare_directories()

    if not 0 <= seed <= 9999:
        raise ValueError("Seed must be an integer from 0000 to 9999.")

    missing = set(MODEL_FEATURES + [YIELD_TARGET, RISK_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.duplicated().any():
        raise ValueError(
            "Remove exact duplicate records before model training."
        )

    validate_feature_names(MODEL_FEATURES)

    X = df[MODEL_FEATURES].copy()
    y_reg = df[YIELD_TARGET].astype(float).copy()
    y_clf = df[RISK_TARGET].astype(int).copy()

    # Split row indices once so both supervised tasks use the same
    # training and test records.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=seed,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    yc_train, yc_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    if yc_train.nunique() < 2:
        raise ValueError(
            "Classification training data must contain both risk classes."
        )

    # ---------------------------------------------------------
    # 1. Regression: baseline and two candidate models
    # ---------------------------------------------------------
    regression_models = {
        "mean_baseline": make_model_pipeline(
            DummyRegressor(strategy="mean")
        ),
        "linear_regression": make_model_pipeline(
            LinearRegression()
        ),
        "random_forest": make_model_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                random_state=seed,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        ),
    }

    regression_metrics = {}

    for name, model in regression_models.items():
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)

        regression_metrics[name] = {
            "mae": float(mean_absolute_error(yr_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yr_test, pred))),
            "r2": float(r2_score(yr_test, pred)),
        }

    selected_regression = min(
        regression_models,
        key=lambda name: regression_metrics[name]["mae"],
    )
    regression_model = regression_models[selected_regression]

    _write_json(
        ARTIFACTS_DIR / "regression_metrics.json",
        {
            "verification_code": f"AI-A2-{seed:04d}",
            "target": YIELD_TARGET,
            "test_rows": int(len(test_idx)),
            "metrics": regression_metrics,
            "selected_model": selected_regression,
            "selection_reason": (
                "Lowest held-out test MAE among the evaluated models."
            ),
        },
    )

    reg_pred = regression_model.predict(X_test)
    residuals = yr_test.to_numpy() - reg_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(reg_pred, residuals, alpha=0.75)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted yield (tons)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Regression Residuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=160)
    plt.close()

    joblib.dump(regression_model, MODELS_DIR / "regression_model.joblib")

    # ---------------------------------------------------------
    # 2. Classification: majority baseline and two candidates
    # ---------------------------------------------------------
    classification_models = {
        "majority_baseline": make_model_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "logistic_regression": make_model_pipeline(
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
            )
        ),
        "random_forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
    }

    classification_metrics = {}

    for name, model in classification_models.items():
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)

        result = {
            "accuracy": float(accuracy_score(yc_test, pred)),
            "precision": float(
                precision_score(yc_test, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(yc_test, pred, zero_division=0)
            ),
            "f1": float(
                f1_score(yc_test, pred, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                yc_test, pred, labels=[0, 1]
            ).tolist(),
        }

        if hasattr(model, "predict_proba") and yc_test.nunique() == 2:
            probabilities = model.predict_proba(X_test)[:, 1]
            result["roc_auc"] = float(
                roc_auc_score(yc_test, probabilities)
            )
        else:
            result["roc_auc"] = None

        classification_metrics[name] = result

    # Prefer the model with the highest F1 score; break ties using
    # ROC-AUC, when available.
    def classification_rank(name: str) -> tuple[float, float]:
        result = classification_metrics[name]
        auc = result["roc_auc"]
        return result["f1"], auc if auc is not None else -1.0

    selected_classifier = max(
        classification_models,
        key=classification_rank,
    )
    classifier_model = classification_models[selected_classifier]

    _write_json(
        ARTIFACTS_DIR / "classification_metrics.json",
        {
            "verification_code": f"AI-A2-{seed:04d}",
            "target": RISK_TARGET,
            "positive_class": 1,
            "test_rows": int(len(test_idx)),
            "metrics": classification_metrics,
            "selected_model": selected_classifier,
            "selection_reason": (
                "Highest held-out test F1; ROC-AUC breaks ties."
            ),
        },
    )

    clf_pred = classifier_model.predict(X_test)
    clf_prob = classifier_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(yc_test, clf_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted low", "Predicted high"])
    ax.set_yticks([0, 1], labels=["Actual low", "Actual high"])
    ax.set_title("High-Risk Confusion Matrix")
    fig.colorbar(display, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    joblib.dump(
        classifier_model,
        MODELS_DIR / "classification_model.joblib",
    )

    # ---------------------------------------------------------
    # 3. Clustering: K-means with K=2 through K=6
    # ---------------------------------------------------------
    # No target columns enter the clustering features.
    cluster_preprocessor = make_preprocessor()
    X_scaled = cluster_preprocessor.fit_transform(X)

    cluster_scores = {}
    cluster_models = {}

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=20,
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        cluster_scores[str(k)] = float(score)
        cluster_models[k] = model

    selected_k = max(
        (int(k) for k in cluster_scores),
        key=lambda k: cluster_scores[str(k)],
    )

    cluster_model = cluster_models[selected_k]
    cluster_labels = cluster_model.labels_

    clustered_df = X.copy()
    clustered_df["cluster"] = cluster_labels

    profiles = {}
    for cluster_id, group in clustered_df.groupby("cluster"):
        profile = {}

        for column in MODEL_FEATURES:
            if column == "season":
                profile[column] = (
                    group[column].mode().iloc[0]
                )
            else:
                profile[column] = float(group[column].mean())

        profile["farm_count"] = int(len(group))
        profiles[str(int(cluster_id))] = profile

    _write_json(
        ARTIFACTS_DIR / "clustering_report.json",
        {
            "verification_code": f"AI-A2-{seed:04d}",
            "features": MODEL_FEATURES,
            "silhouette_scores": cluster_scores,
            "selected_k": selected_k,
            "selection_reason": (
                "Highest silhouette score among K=2 through K=6."
            ),
            "cluster_profiles": profiles,
        },
    )

    # Save cluster assignments for the held-out test records.
    test_cluster_labels = cluster_model.predict(
        cluster_preprocessor.transform(X_test)
    )

    # Visualize clusters in two PCA dimensions.
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.8,
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"Farm Clusters (K={selected_k})")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clusters.png", dpi=160)
    plt.close()

    joblib.dump(
        {
            "preprocessor": cluster_preprocessor,
            "model": cluster_model,
            "features": MODEL_FEATURES,
        },
        MODELS_DIR / "clustering_model.joblib",
    )

    # ---------------------------------------------------------
    # 4. Consolidated held-out predictions
    # ---------------------------------------------------------
    predictions = pd.DataFrame(
        {
            "RowID": test_idx,
            "actual_yield_tons": yr_test.to_numpy(),
            "predicted_yield_tons": reg_pred,
            "yield_error": residuals,
            "absolute_yield_error": np.abs(residuals),
            "actual_high_risk": yc_test.to_numpy(),
            "predicted_high_risk": clf_pred,
            "high_risk_probability": clf_prob,
            "cluster": test_cluster_labels,
        }
    )

    predictions.to_csv(
        ARTIFACTS_DIR / "test_predictions.csv",
        index=False,
        float_format="%.4f",
    )

    summary = {
        "verification_code": f"AI-A2-{seed:04d}",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "selected_regression_model": selected_regression,
        "selected_classification_model": selected_classifier,
        "selected_cluster_count": selected_k,
    }

    _write_json(ARTIFACTS_DIR / "training_summary.json", summary)

    return summary
