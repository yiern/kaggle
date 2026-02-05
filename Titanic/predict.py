#!/usr/bin/env python3
"""XGBoost prediction script for Titanic survival prediction."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.preprocessing.feature_engineering import preprocess

# Config
DATA_DIR = Path("./titanic")
MODEL_PATH = Path("./models/xgboost.pkl")


def load_test_data():
    """Load and preprocess test data."""
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df = preprocess(test_df)

    ids = test_df["PassengerId"]
    X = test_df.drop(["PassengerId"], axis=1)

    return X, ids


def predict():
    """Generate predictions on test set."""
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load test data
    X_test, ids = load_test_data()

    # Predict
    predictions = model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({"PassengerId": ids, "Survived": predictions})
    submission.to_csv("submission.csv", index=False)
    print(f"Predictions saved: submission.csv ({len(submission)} rows)")


if __name__ == "__main__":
    predict()
