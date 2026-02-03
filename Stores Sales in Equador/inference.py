#!/usr/bin/env python3
"""
Generate Kaggle submission predictions using trained Meta-Learning model (v4.1a).

This script loads a trained MetaLearningMultiTaskLSTM model and generates predictions
for the Kaggle test set using 3D hierarchical tensors.

Model Version: v4.1a (Meta-Learning Multi-Task LSTM + Proper Encoding)
Features: ~30-32 features (dynamic, based on one-hot encoding)
  - Base: onpromotion
  - Temporal: day, month, quarter, year, day_of_week, is_weekend
  - External: oil_price
  - Lag: sale_lag (1 day)
  - Rolling: mean_7, std_7, mean_30, min_7, max_7
  - Holiday: is_holiday, is_national_holiday, holiday_type (one-hot: 6 features)
  - Store: store_type (one-hot: 5 features), cluster, city (target encoded), state (target encoded)

Architecture:
  - Family encoder: Embeds 33 families into 16-dim vectors
  - Shared LSTM: Learns temporal patterns across all stores
  - Store adapters: 54 store-specific adapters for fine-tuning
  - Family decoder: Projects to 33 family predictions simultaneously

Usage:
    python inference.py

Requirements:
    - models/best_model_multitask.pt must exist (run train_model.py first)
    - data/preprocessing_metadata.pkl must exist (created by create_processed_data.py)
    - Kaggle dataset files in ./store-sales-time-series-forecasting/

Note:
    - Target variable uses log1p transform (applied during preprocessing)
    - Model outputs are inverse transformed using np.expm1() for submission
    - Inference uses EXACT same preprocessing as training (via saved metadata)
"""

import pandas as pd
import torch
import numpy as np
import sys
import os
import joblib
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ML_Models.multitask_lstm import MetaLearningMultiTaskLSTM, RMSLELoss
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, log_file="logs/inference.log")


def load_trained_model(checkpoint_path: str, device: str):
    """
    Load trained Meta-Learning model from checkpoint with validation.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: 'mps', 'cuda', or 'cpu'

    Returns:
        model: Loaded MetaLearningMultiTaskLSTM in eval mode
        metadata: Dict with model version info

    Raises:
        ValueError: If checkpoint is missing required metadata or version doesn't match
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint file is corrupt or incompatible
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    # Validate checkpoint file exists
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(
            f"❌ Checkpoint file not found: {checkpoint_path}\n"
            f"Please train the model first or verify the path."
        )

    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        logger.info("✓ Checkpoint loaded successfully")
    except (RuntimeError, EOFError) as e:
        raise RuntimeError(
            f"❌ Failed to load checkpoint (file may be corrupt): {checkpoint_path}\n"
            f"Error: {str(e)}\n"
            f"Please retrain the model or use a different checkpoint."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"❌ Unexpected error loading checkpoint: {checkpoint_path}\n"
            f"Error: {type(e).__name__}: {str(e)}"
        ) from e

    # Validate required metadata
    required_keys = ["num_features", "num_families", "num_stores", "model_type"]
    missing_keys = [key for key in required_keys if key not in checkpoint]

    if missing_keys:
        raise ValueError(
            f"❌ Checkpoint missing required metadata: {missing_keys}\n"
            f"Available keys: {list(checkpoint.keys())}\n"
            f"Please train the model with updated code that saves all dimensions."
        )

    # Extract dimensions from checkpoint
    num_features = checkpoint["num_features"]
    num_families = checkpoint["num_families"]
    num_stores = checkpoint["num_stores"]
    model_type = checkpoint["model_type"]
    epoch = checkpoint.get("epoch", 0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    logger.info("=" * 70)
    logger.info("CHECKPOINT INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Model Type:       {model_type}")
    logger.info("Dimensions (from checkpoint):")
    logger.info(f"  Features:       {num_features}")
    logger.info(f"  Families:       {num_families}")
    logger.info(f"  Stores:         {num_stores}")
    logger.info(f"Epoch Trained:    {epoch}")
    logger.info(f"Best Val Loss:    {best_val_loss:.6f}")
    logger.info("=" * 70)

    # Validate model type
    if model_type != "multitask_lstm":
        raise ValueError(
            f"Model type mismatch!\n"
            f"  Expected: multitask_lstm\n"
            f"  Got:      {model_type}\n"
            f"  This inference script is for Meta-Learning models only."
        )

    # Recreate model with dimensions from checkpoint
    model = MetaLearningMultiTaskLSTM(
        num_features=num_features,
        num_families=num_families,
        num_stores=num_stores,
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"✓ Model loaded (epoch {epoch})")
    logger.info(
        f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    metadata = {
        "model_type": model_type,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "feature_cols": checkpoint.get("feature_cols", []),
        "num_features": num_features,
        "num_families": num_families,
        "num_stores": num_stores,
    }

    return model, metadata


def preprocess_for_inference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oil_data: pd.DataFrame,
    stores_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    preprocessing_metadata: dict,
    scaler_X_path: Path,
):
    """
    Preprocess train and test data for inference with v4.1a model.

    This function replicates the EXACT same preprocessing used during training
    by loading and using the saved preprocessing_metadata.

    Args:
        train_df: Training data with date index
        test_df: Test data with date index
        oil_data: Oil price data with date index
        stores_df: Store information data
        holidays_df: Holidays events data
        preprocessing_metadata: Saved metadata from training (city/state means, column names)
        scaler_X_path: Path to saved scaler_X.pkl

    Returns:
        train_proc: Processed training data (with log1p(sales))
        test_proc: Processed test data
    """
    logger.info("Preprocessing data for v4.1a inference...")
    logger.info("  Using saved preprocessing metadata for consistency")

    # Load preprocessing metadata
    city_mean_sales = preprocessing_metadata["city_mean_sales"]
    state_mean_sales = preprocessing_metadata["state_mean_sales"]
    one_hot_cols = preprocessing_metadata["one_hot_cols"]
    feature_cols = preprocessing_metadata["feature_cols"]
    base_feature_cols = preprocessing_metadata["base_feature_cols"]

    # Merge external data (stores and holidays)
    logger.info("  Merging external data...")
    train_df = train_df.merge(stores_df, on="store_nbr", how="left")
    test_df = test_df.merge(stores_df, on="store_nbr", how="left")

    train_df = train_df.merge(holidays_df, on="date", how="left")
    test_df = test_df.merge(holidays_df, on="date", how="left")

    train_df = train_df.merge(oil_data, on="date", how="left")
    test_df = test_df.merge(oil_data, on="date", how="left")

    # Fill missing values
    train_df["dcoilwtico"] = train_df["dcoilwtico"].ffill().fillna(0)
    test_df["dcoilwtico"] = test_df["dcoilwtico"].ffill().fillna(0)
    train_df["dcoilwtico"] = train_df["dcoilwtico"].fillna(0)
    test_df["dcoilwtico"] = test_df["dcoilwtico"].fillna(0)

    # Rename oil column
    train_df = train_df.rename(columns={"dcoilwtico": "oil_price"})
    test_df = test_df.rename(columns={"dcoilwtico": "oil_price"})

    # Encode family (string to integer)
    family_encoder = {
        family: idx for idx, family in enumerate(sorted(train_df["family"].unique()))
    }
    train_df["family"] = train_df["family"].map(lambda x: family_encoder.get(x, -1))
    test_df["family"] = test_df["family"].map(lambda x: family_encoder.get(x, -1))

    # NOTE: item_enc is NOT used in v4.0+ Meta-Learning architecture
    # The v4.0+ model uses Family Encoder + Store Adapters instead
    # item_enc column will be automatically excluded by dataloader (data_multitask.py line 89)

    # Convert date to datetime
    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # Temporal features
    for df in [train_df, test_df]:
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holiday features
    print("  Creating holiday features...")
    for df in [train_df, test_df]:
        df["is_holiday"] = df["type_y"].notna().astype(int)
        df["is_national_holiday"] = (df["locale"] == "National").astype(int)
        df["holiday_type"] = df["type_y"].fillna("None")

    # Store features
    print("  Creating store features...")
    for df in [train_df, test_df]:
        df["store_type"] = df["type_x"]  # Renamed from stores merge
        df["cluster"] = df["cluster"]

    # Drop original type columns (after creating store_type)
    train_df = train_df.drop(
        columns=["type_x", "locale", "locale_name", "description", "transferred"]
    )
    test_df = test_df.drop(
        columns=["type_x", "locale", "locale_name", "description", "transferred"]
    )

    # TARGET ENCODING: Use saved means from training (CRITICAL!)
    print("  Applying target encoding (using saved means from training)...")
    train_df["city_encoded"] = train_df["city"].map(city_mean_sales)
    test_df["city_encoded"] = (
        test_df["city"].map(city_mean_sales).fillna(city_mean_sales.mean())
    )

    train_df["state_encoded"] = train_df["state"].map(state_mean_sales)
    test_df["state_encoded"] = (
        test_df["state"].map(state_mean_sales).fillna(state_mean_sales.mean())
    )

    # ONE-HOT ENCODING: Create same columns as training
    print("  Applying one-hot encoding...")
    train_df = pd.get_dummies(
        train_df, columns=["holiday_type"], prefix="holiday", dtype=float
    )
    test_df = pd.get_dummies(
        test_df, columns=["holiday_type"], prefix="holiday", dtype=float
    )

    train_df = pd.get_dummies(
        train_df, columns=["store_type"], prefix="store_type", dtype=float
    )
    test_df = pd.get_dummies(
        test_df, columns=["store_type"], prefix="store_type", dtype=float
    )

    # Ensure all one-hot columns exist (test might be missing some)
    for col in one_hot_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    # Keep only the one-hot columns that exist in training
    test_df = test_df[[col for col in test_df.columns if col in train_df.columns]]

    # Drop original categorical columns (now encoded)
    train_df = train_df.drop(columns=["city", "state"])
    test_df = test_df.drop(columns=["city", "state"])

    # Lag features
    train_df["sale_lag"] = train_df.groupby(["store_nbr", "family"])["sales"].shift(1)
    train_df["sale_lag"] = train_df["sale_lag"].fillna(0)

    test_df = test_df.sort_values(["store_nbr", "family", "date"])
    train_last_sales = (
        train_df.groupby(["store_nbr", "family"])["sales"].last().reset_index()
    )
    train_last_sales = train_last_sales.rename(columns={"sales": "sale_lag"})
    test_df = test_df.merge(train_last_sales, on=["store_nbr", "family"], how="left")
    test_df["sale_lag"] = test_df["sale_lag"].fillna(0)

    # Rolling features
    for window in [7, 30]:
        train_df[f"rolling_mean_{window}"] = train_df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        train_df[f"rolling_std_{window}"] = train_df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())

    rolling_cols = [c for c in train_df.columns if "rolling_" in c]
    train_df[rolling_cols] = train_df[rolling_cols].fillna(0)

    # Min/max rolling
    train_df["rolling_min_7"] = train_df.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).min())
    train_df["rolling_max_7"] = train_df.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).max())
    train_df[["rolling_min_7", "rolling_max_7"]] = train_df[
        ["rolling_min_7", "rolling_max_7"]
    ].fillna(0)

    # For test data, use last rolling values from training
    train_df = train_df.sort_values(["store_nbr", "family", "date"])
    for window in [7, 30]:
        train_last_rolling = (
            train_df.groupby(["store_nbr", "family"])[
                [f"rolling_mean_{window}", f"rolling_std_{window}"]
            ]
            .last()
            .reset_index()
        )
        test_df = test_df.merge(
            train_last_rolling,
            on=["store_nbr", "family"],
            how="left",
            suffixes=("", "_train"),
        )
        test_df[f"rolling_mean_{window}"] = (
            test_df[f"rolling_mean_{window}"]
            .fillna(test_df.get(f"rolling_mean_{window}_train", 0))
            .fillna(0)
        )
        test_df[f"rolling_std_{window}"] = (
            test_df[f"rolling_std_{window}"]
            .fillna(test_df.get(f"rolling_std_{window}_train", 0))
            .fillna(0)
        )

    train_last_minmax = (
        train_df.groupby(["store_nbr", "family"])[["rolling_min_7", "rolling_max_7"]]
        .last()
        .reset_index()
    )
    test_df = test_df.merge(
        train_last_minmax,
        on=["store_nbr", "family"],
        how="left",
        suffixes=("", "_train"),
    )
    test_df["rolling_min_7"] = (
        test_df["rolling_min_7"].fillna(test_df.get("rolling_min_7_train", 0)).fillna(0)
    )
    test_df["rolling_max_7"] = (
        test_df["rolling_max_7"].fillna(test_df.get("rolling_max_7_train", 0)).fillna(0)
    )

    # Load scaler and scale features
    print("  Loading scaler and scaling features...")
    scaler_X = joblib.load(scaler_X_path)

    # Scale all features (scaler was fitted on all feature_cols including one-hot)
    train_df[feature_cols] = scaler_X.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler_X.transform(test_df[feature_cols])

    # Apply log1p transform to target (same as training)
    print("  Applying log1p transform to target...")
    train_df["sales"] = np.log1p(train_df["sales"])

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    # Keep only feature columns + metadata
    metadata_cols = ["id", "date", "store_nbr", "family"]
    train_proc = train_df[metadata_cols + feature_cols + ["sales"]]
    test_proc = test_df[metadata_cols + feature_cols]

    print(f"  ✓ Preprocessing complete")
    print(f"    Feature count: {len(feature_cols)}")
    print(f"    One-hot columns: {len(one_hot_cols)}")

    return train_proc, test_proc


def create_3d_test_tensors(
    train_proc: pd.DataFrame,
    test_proc: pd.DataFrame,
    store_id: int,
    window_size: int = 7,
    preprocessing_metadata: dict = None,
):
    """
    Create 3D test tensors for a specific store using dynamic feature detection.

    Args:
        train_proc: Processed training data
        test_proc: Processed test data
        store_id: Store ID to create tensors for
        window_size: Number of days in input window (default 7)
        preprocessing_metadata: Metadata from training containing feature_cols

    Returns:
        test_tensors: List of 3D tensors [1, Window-1, Features, Families]
        test_dates: List of test dates
        family_ids: List of family IDs in the test set
    """
    # Filter data for this store
    train_store = train_proc[train_proc["store_nbr"] == store_id]
    test_store = test_proc[test_proc["store_nbr"] == store_id]

    # Set date as index for time series operations
    train_store = train_store.set_index("date")
    test_store = test_store.set_index("date")

    # Get unique families
    family_ids = sorted(test_store["family"].unique())

    # Get feature columns from preprocessing metadata (dynamic!)
    if preprocessing_metadata is not None:
        feature_cols = preprocessing_metadata["feature_cols"]
        print(f"    Using {len(feature_cols)} features from metadata")
    else:
        # Fallback: use all columns except metadata and target
        exclude_cols = ["id", "date", "store_nbr", "family", "sales"]
        feature_cols = [col for col in train_proc.columns if col not in exclude_cols]
        print(
            f"    WARNING: Using detected features ({len(feature_cols)}): {feature_cols[:5]}..."
        )

    # Get unique test dates
    test_dates = test_store.index.unique()

    test_tensors = []

    for test_date in test_dates:
        # For each test date, get (window_size - 1) days before it
        historical_end = pd.Timestamp(test_date)
        historical_start = historical_end - pd.Timedelta(days=window_size - 2)

        # Get historical data for all families
        # Use boolean indexing instead of .loc[] because index has duplicates (multiple families per date)
        historical_data = train_store[
            (train_store.index >= historical_start)
            & (train_store.index <= historical_end)
        ]

        # Check if we have enough data
        if len(historical_data) < window_size - 1:
            # Pad with zeros if not enough data
            padding_size = (window_size - 1) - len(historical_data)
            pad_start = historical_start - pd.Timedelta(days=padding_size - 1)
            padding_df = pd.DataFrame(
                0,
                index=pd.date_range(start=pad_start, periods=padding_size, freq="D"),
                columns=historical_data.columns,
            )
            historical_data = pd.concat([padding_df, historical_data])

        # Pivot to create 3D tensor: [Dates, Families, Features]
        pivot_data = historical_data.pivot_table(
            index=historical_data.index,
            columns="family",
            values=feature_cols,
            aggfunc="mean",
        )

        # Build complete column structure for all features × all families
        # Reindex handles missing families and fills with 0 (no fragmentation)
        expected_columns = [
            (col, fam_id) for col in feature_cols for fam_id in family_ids
        ]
        pivot_data = pivot_data.reindex(columns=expected_columns, fill_value=0)

        # CRITICAL FIX: Ensure we have exactly window_size - 1 rows
        # pivot_table may produce fewer rows if dates are missing
        complete_date_range = pd.date_range(
            start=historical_start, end=historical_end, freq="D"
        )
        pivot_data = pivot_data.reindex(complete_date_range, fill_value=0)
        # Take the last window_size - 1 rows to ensure correct shape
        pivot_data = pivot_data.iloc[-(window_size - 1) :]

        # Convert to tensor: [Dates, Features, Families]
        tensor_data = pivot_data.to_numpy().reshape(
            window_size - 1, len(feature_cols), len(family_ids)
        )

        # Convert to torch tensor: [1, Window-1, Features, Families]
        tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)

        test_tensors.append(tensor)

    return test_tensors, test_dates, family_ids


def generate_predictions(
    model,
    train_proc: pd.DataFrame,
    test_proc: pd.DataFrame,
    device: str,
    family_encoder: dict,
    preprocessing_metadata: dict = None,
    checkpoint_metadata: dict = None,
):
    """
    Generate predictions for all stores using 3D multitask model.

    Args:
        model: Trained MetaLearningMultiTaskLSTM
        train_proc: Processed training data
        test_proc: Processed test data
        device: 'mps', 'cuda', or 'cpu'
        family_encoder: Dict mapping family names to IDs
        preprocessing_metadata: Metadata from training (for dynamic feature detection)

    Returns:
        List of predictions, where each is:
        {
            'id': test ID,
            'sales': predicted sales (original scale, non-negative)
        }
    """
    model.eval()
    results = []

    # Get unique stores
    store_ids = sorted(test_proc["store_nbr"].unique())

    print(f"Generating predictions for {len(store_ids)} stores...")

    for store_id in store_ids:
        print(f"  Processing store {store_id}/{max(store_ids)}...")

        # Filter test data for this store
        test_store = test_proc[test_proc["store_nbr"] == store_id]

        # Check if store has test data
        if len(test_store) == 0:
            print(f"    WARNING: No test data for store {store_id}, skipping...")
            continue

        # Create 3D test tensors for this store
        test_tensors, test_dates, family_ids = create_3d_test_tensors(
            train_proc,
            test_proc,
            store_id,
            window_size=checkpoint_metadata.get("window_size", 7),
            preprocessing_metadata=preprocessing_metadata,
        )

        if len(test_tensors) == 0:
            continue

        # Generate predictions for each test date
        with torch.no_grad():
            for tensor, test_date in zip(test_tensors, test_dates):
                tensor = tensor.to(device)

                # Forward pass with store adapter
                predictions = model(tensor, store_nbr=store_id)  # Shape: [1, 33]

                # Convert to numpy
                pred_sales = predictions.cpu().detach().numpy().ravel()  # Shape: [33]

                # Ensure non-negative and apply log1p inverse
                pred_sales = np.maximum(0, np.expm1(pred_sales))

                # Get test data for this store and date
                test_store_date = test_proc[
                    (test_proc["store_nbr"] == store_id)
                    & (test_proc["date"] == test_date)
                ]

                # Create results for each family
                for family_id, sales in zip(family_ids, pred_sales):
                    family_name = (
                        family_id  # Already a string family name, no encoding needed
                    )

                    # Get the original test ID
                    test_row = test_store_date[test_store_date["family"] == family_id]

                    if len(test_row) > 0:
                        # Convert to int directly (single row DataFrame)
                        test_id = int(test_row["id"].tolist()[0])

                        results.append({"id": test_id, "sales": sales})

    print(f"✓ Generated {len(results)} predictions")

    return results


def main():
    """Main inference script - Loads model, processes data, generates predictions."""
    logger.info("=" * 60)
    logger.info("INFERENCE SCRIPT: Meta-Learning Multi-Task LSTM")
    logger.info("=" * 60)
    logger.info("Stores Sales Forecasting - Kaggle Competition")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Meta-Learning Multi-Task LSTM + Proper Encoding")
    logger.info("=" * 60)

    # Load trained model

    checkpoint_path = "./models/best_model_multitask.pt"
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint_metadata = load_trained_model(str(checkpoint_path), device)
    model = model.to(device)

    # Load data with error handling
    logger.info("")
    logger.info("Loading data...")

    try:
        train_df = pd.read_csv(
            "./store-sales-time-series-forecasting/train.csv", parse_dates=["date"]
        )
        test_df = pd.read_csv(
            "./store-sales-time-series-forecasting/test.csv", parse_dates=["date"]
        )
        logger.info(f"  ✓ Loaded {len(train_df)} training rows")
        logger.info(f"  ✓ Loaded {len(test_df)} test rows")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ Required data file not found: {str(e)}\n"
            f"Please ensure train.csv and test.csv are in ./store-sales-time-series-forecasting/"
        ) from e
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load data: {str(e)}") from e

    # Load external data
    logger.info("  Loading external data...")

    try:
        oil_data = pd.read_csv(
            "./store-sales-time-series-forecasting/oil.csv", parse_dates=["date"]
        )
        stores_df = pd.read_csv("./store-sales-time-series-forecasting/stores.csv")
        holidays_df = pd.read_csv(
            "./store-sales-time-series-forecasting/holidays_events.csv",
            parse_dates=["date"],
        )
        logger.info(f"  ✓ Loaded {len(oil_data)} oil records")
        logger.info(f"  ✓ Loaded {len(stores_df)} stores")
        logger.info(f"  ✓ Loaded {len(holidays_df)} holiday events")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ External data file not found: {str(e)}\n"
            f"Please ensure oil.csv, stores.csv, and holidays_events.csv are in ./store-sales-time-series-forecasting/"
        ) from e
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load external data: {str(e)}") from e

    # Load preprocessing metadata (CRITICAL for consistency!)
    logger.info("  Loading preprocessing metadata...")
    preprocessing_metadata_path = "./data/preprocessing_metadata.pkl"
    if not Path(preprocessing_metadata_path).exists():
        raise FileNotFoundError(
            f"❌ Missing preprocessing_metadata.pkl! "
            f"Run create_processed_data.py first to ensure training/inference consistency."
        )

    try:
        preprocessing_metadata = joblib.load(preprocessing_metadata_path)
        logger.info(
            f"    ✓ Loaded metadata for {preprocessing_metadata['feature_count']} features"
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load preprocessing metadata: {str(e)}") from e

    # ===== CRITICAL VALIDATION: Feature Name Match =====
    logger.info("")
    logger.info("Validating feature consistency...")
    checkpoint_features = set(checkpoint_metadata.get("feature_cols", set()))
    metadata_features = set(preprocessing_metadata["feature_cols"])

    if checkpoint_features != metadata_features:
        missing_in_metadata = checkpoint_features - metadata_features
        extra_in_metadata = metadata_features - checkpoint_features

        logger.error("❌ CRITICAL: Feature name mismatch detected!")
        logger.error(f"   Checkpoint expects: {len(checkpoint_features)} features")
        logger.error(f"   Preprocessing has:  {len(metadata_features)} features")

        if missing_in_metadata:
            logger.error(f"   Missing in metadata: {missing_in_metadata}")
        if extra_in_metadata:
            logger.error(f"   Extra in metadata:  {extra_in_metadata}")

        raise ValueError(
            "Feature name mismatch between checkpoint and preprocessing metadata!\n"
            "This will cause incorrect predictions or shape errors.\n"
            "Ensure you're using the correct preprocessing_metadata.pkl file "
            "that was created during the same training run."
        )
    else:
        logger.info(f"    ✓ All {len(metadata_features)} feature names match")

    # Preprocess (with v4.1a support)
    logger.info("")
    logger.info("Preprocessing data...")
    train_proc, test_proc = preprocess_for_inference(
        train_df=train_df,
        test_df=test_df,
        oil_data=oil_data,
        stores_df=stores_df,
        holidays_df=holidays_df,
        preprocessing_metadata=preprocessing_metadata,
        scaler_X_path=Path("./data/scaler_X.pkl"),
    )

    # Generate predictions (with preprocessing metadata)
    logger.info("")
    logger.info("Generating predictions...")
    results = generate_predictions(
        model=model,
        train_proc=train_proc,
        test_proc=test_proc,
        device=device,
        family_encoder={},  # Empty dict, not needed for v4.1a
        preprocessing_metadata=preprocessing_metadata,
        checkpoint_metadata=checkpoint_metadata,
    )

    # ===== Step 6: Create submission DataFrame =====
    submission = pd.DataFrame(results)
    submission["id"] = submission["id"].astype("int32")

    # ===== Step 7: Validate submission =====
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATING SUBMISSION")
    logger.info("=" * 70)

    # Check shape
    expected_rows = len(test_df)
    actual_rows = len(submission)
    assert actual_rows == expected_rows, (
        f"Wrong shape! Expected {expected_rows}, got {actual_rows}"
    )
    logger.info(f"✓ Shape correct: {submission.shape}")

    # Check for NaN
    nan_count = submission["sales"].isna().sum()
    assert nan_count == 0, f"Submission contains {nan_count} NaN values!"
    logger.info(f"✓ No NaN values")

    # Check non-negative
    negative_count = (submission["sales"] < 0).sum()
    assert negative_count == 0, (
        f"Submission contains {negative_count} negative predictions!"
    )
    logger.info(f"✓ All predictions non-negative")

    # ===== Step 8: Save submission =====
    submission_file = "submission.csv"
    submission.to_csv(submission_file, index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUBMISSION CREATED!")
    logger.info("=" * 70)
    logger.info(f"File: {submission_file}")
    logger.info("")
    logger.info("Sample predictions:")
    logger.info(f"\n{submission.head(10)}")
    logger.info("")
    logger.info("📊 Prediction Statistics:")
    logger.info(f"  Mean: {submission['sales'].mean():.2f}")
    logger.info(f"  Min: {submission['sales'].min():.2f}")
    logger.info(f"  Max: {submission['sales'].max():.2f}")
    logger.info(f"  Std: {submission['sales'].std():.2f}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✅ Ready to submit to Kaggle!")
    logger.info("   Command:")
    logger.info(
        "   kaggle competitions submit -c store-sales-time-series-forecasting \\"
    )
    logger.info(f"     -f {submission_file} \\")
    logger.info(f"     -m 'Meta-Learning Multi-Task LSTM'")


if __name__ == "__main__":
    main()
