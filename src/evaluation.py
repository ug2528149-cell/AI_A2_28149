import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def evaluate_regression(y_true, y_pred):
    """Return regression metrics on held-out observations."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true, y_pred, y_probability=None):
    """Return classification metrics and confusion matrix."""
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "roc_auc": None,
    }

    if y_probability is not None and len(set(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, y_probability)
        )

    return result