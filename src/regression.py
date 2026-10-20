import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import (
        CATEGORICAL_FEATURES,
        DATA_FILE,
        RANDOM_STATE,
        REGRESSION_TARGET,
        TEST_SIZE,
    )
    from .evaluation import evaluate_regression
except ImportError:
    from config import (
        CATEGORICAL_FEATURES,
        DATA_FILE,
        RANDOM_STATE,
        REGRESSION_TARGET,
        TEST_SIZE,
    )
    from evaluation import evaluate_regression


def train_regression_model(data=None):
    """Train and evaluate a linear regression yield model."""
    if data is None:
        data = pd.read_csv(DATA_FILE)

    data = data.drop_duplicates().copy()

    # The target must exist before training.
    if REGRESSION_TARGET not in data.columns:
        raise ValueError(
            f"Missing target column: {REGRESSION_TARGET}"
        )

    # Never use either target as a predictor.
    excluded = [REGRESSION_TARGET, "high_risk"]
    feature_columns = [
        column for column in data.columns
        if column not in excluded
    ]

    data = data.dropna(subset=[REGRESSION_TARGET])

    X = data[feature_columns]
    y = data[REGRESSION_TARGET]

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "metrics": evaluate_regression(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    result = train_regression_model()

    print("REGRESSION MODEL EVALUATION")
    print(result["metrics"])
    print("Training rows:", result["train_rows"])
    print("Testing rows:", result["test_rows"])