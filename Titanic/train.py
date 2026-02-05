#!/usr/bin/env python3
"""XGBoost training with Optuna Bayesian optimization."""

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from src.preprocessing.feature_engineering import preprocess
from src.evaluation.metrics import plot_feature_importance

import optuna
from optuna.samplers import TPESampler

# Config
DATA_DIR = Path("./titanic")
MODELS_DIR = Path("./models")
OUTPUTS_DIR = Path("./outputs")
MODEL_PATH = MODELS_DIR / "xgboost.pkl"
RANDOM_STATE = 42
CV_FOLDS = 10
TEST_SIZE = 0.2
EARLY_STOPPING_ROUNDS = 20
N_TRIALS = 100


def load_train_data():
    """Load and preprocess training data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df = preprocess(train_df)

    y = train_df["Survived"]
    X = train_df.drop(["Survived", "PassengerId"], axis=1)

    return X, y


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    return scores.mean()


def train():
    """Train XGBoost model."""
    X, y = load_train_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline CV
    baseline = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    )
    scores = cross_val_score(
        baseline, X_train, y_train, cv=CV_FOLDS, scoring="accuracy"
    )
    print(f"Baseline CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Optuna optimization
    print(f"Optimizing... ({N_TRIALS} trials)")

    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=-1,
    )

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed / 60:.1f} minutes")

    # Results
    print(f"\nBest CV Accuracy: {study.best_value:.4f}")
    print(f"Best Params:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Train final model
    best = XGBClassifier(**study.best_params)
    best.fit(X_train, y_train)

    train_acc = best.score(X_train, y_train)
    val_acc = best.score(X_val, y_val)
    print(f"\nTrain: {train_acc:.4f}, Val: {val_acc:.4f}")

    # Feature importance
    importance = best.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": [importance.get(f"f{i}", 0) for i in range(len(X.columns))],
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 5 Features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:<15} {row['importance']:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    plot_feature_importance(
        importance_df, save_path=OUTPUTS_DIR / "feature_importance.png"
    )
    print(f"Plot saved: {OUTPUTS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    train()
