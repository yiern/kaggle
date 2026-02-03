"""
Training utilities and checkpoint management.
"""

import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ML_Models.multitask_lstm import RMSLELoss


# Model version constants
MODEL_VERSION = "v4.1e"
MODEL_DESCRIPTION = "Meta-Learning Multi-Task LSTM with 3D hierarchical tensors + holiday & store features + log-transformed sales-dependent features (fixed scale mismatch)"
FEATURE_COUNT = (
    32  # Number of numerical features (excluding family, item_enc, and store_nbr)
    # 14 original + 11 new (3 holiday + 2 store_type one-hot + 2 target encoded city/state + 6 holiday one-hot)
)


def engineer_features_and_preprocess(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    oil_data: pd.DataFrame,
    stores_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, int]:
    """
    Engineer all temporal features and preprocess data for meta-learning model.

    This function creates the 21 numerical features required by the v4.1 model:
    - Base: onpromotion
    - Temporal: day, month, quarter, year, day_of_week, is_weekend
    - External: oil_price
    - Lag: sale_lag (1-day)
    - Rolling: mean_7, std_7, mean_30, min_7, max_7
    - Holiday: is_holiday, is_national_holiday, holiday_type (NEW)
    - Store: store_type, store_cluster, city, state (NEW)

    Args:
        train_df: Training DataFrame with date index and 'sales' column
        val_df: Validation DataFrame with date index and 'sales' column
        oil_data: Oil price DataFrame with date index
        stores_df: Store metadata DataFrame
        holidays_df: Holidays events DataFrame

    Returns:
        train_proc: Processed training DataFrame (sales log1p-transformed)
        val_proc: Processed validation DataFrame (sales log1p-transformed)
        scaler_X: Fitted StandardScaler for features
        feature_count: Number of features (should be 21)

    Note:
        Target variable (sales) is transformed using log1p() to match Kaggle RMSLE metric.
        Use np.expm1() to inverse transform predictions.
    """
    print("  Engineering features...")

    # Ensure date is named as 'date' in index
    train_df.index.name = "date"
    val_df.index.name = "date"

    # Reset index to make date a column for merging
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    oil_data = oil_data.reset_index()

    # Merge oil data
    train_df = train_df.merge(oil_data, on="date", how="left")
    val_df = val_df.merge(oil_data, on="date", how="left")

    # Rename oil column
    train_df = train_df.rename(columns={"dcoilwtico": "oil_price"})
    val_df = val_df.rename(columns={"dcoilwtico": "oil_price"})

    # Fill missing oil prices
    train_df["oil_price"] = train_df["oil_price"].ffill().fillna(0)
    val_df["oil_price"] = val_df["oil_price"].ffill().fillna(0)

    # ===== NEW: Merge holiday features =====
    print("  Adding holiday features...")
    # Prepare holidays data
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    holidays_df = holidays_df[
        holidays_df["transferred"] != True
    ]  # Exclude transferred holidays

    # Create holiday indicators
    holiday_features = (
        holidays_df.groupby("date")
        .agg(
            {
                "type": "first",  # Holiday type
                "locale": lambda x: "National" in x.values,  # Is national holiday
            }
        )
        .reset_index()
    )
    holiday_features.columns = ["date", "holiday_type", "is_national_holiday"]
    holiday_features["is_holiday"] = 1  # All rows in this df are holidays

    # Merge holiday features
    train_df = pd.merge(train_df, holiday_features, on="date", how="left")
    val_df = pd.merge(val_df, holiday_features, on="date", how="left")

    # Fill missing holiday values (non-holiday days)
    for col in ["is_holiday", "is_national_holiday", "holiday_type"]:
        train_df[col] = train_df[col].fillna(0 if col != "holiday_type" else "None")
        val_df[col] = val_df[col].fillna(0 if col != "holiday_type" else "None")

    # ===== NEW: Merge store metadata =====
    print("  Adding store features...")
    # Merge store features
    train_df = pd.merge(train_df, stores_df, on="store_nbr", how="left")
    val_df = pd.merge(val_df, stores_df, on="store_nbr", how="left")

    # ===== NEW: Encode categorical features BEFORE target transform =====
    print("  Encoding categorical features...")
    print("    (Target encoding city/state using RAW sales values)")

    # 1. Target encoding for MEDIUM cardinality features (city, state)
    # MUST be done on RAW sales before log1p transform!
    city_mean_sales = train_df.groupby("city")["sales"].mean()
    state_mean_sales = train_df.groupby("state")["sales"].mean()

    train_df["city_encoded"] = train_df["city"].map(city_mean_sales)
    val_df["city_encoded"] = (
        val_df["city"].map(city_mean_sales).fillna(city_mean_sales.mean())
    )

    train_df["state_encoded"] = train_df["state"].map(state_mean_sales)
    val_df["state_encoded"] = (
        val_df["state"].map(state_mean_sales).fillna(state_mean_sales.mean())
    )

    # Drop original categorical columns (now encoded)
    train_df = train_df.drop(columns=["city", "state"])
    val_df = val_df.drop(columns=["city", "state"])

    # NOTE: item_enc is NOT used in v4.0+ Meta-Learning architecture
    # v4.0 uses Family Encoder + Store Adapters instead
    # item_enc is kept in data for compatibility but excluded from features (see data_multitask.py line 89)

    # 2. One-hot encoding for LOW cardinality features (holiday_type, store_type)
    print("    One-hot encoding holiday_type and store_type...")
    train_df = pd.get_dummies(
        train_df, columns=["holiday_type"], prefix="holiday", dtype=float
    )
    val_df = pd.get_dummies(
        val_df, columns=["holiday_type"], prefix="holiday", dtype=float
    )

    train_df = pd.get_dummies(
        train_df, columns=["type"], prefix="store_type", dtype=float
    )
    val_df = pd.get_dummies(val_df, columns=["type"], prefix="store_type", dtype=float)

    # Ensure both train and val have same columns (in case val is missing a category)
    one_hot_cols = [
        c
        for c in train_df.columns
        if c.startswith("holiday_") or c.startswith("store_type_")
    ]
    for col in one_hot_cols:
        if col not in val_df.columns:
            val_df[col] = 0.0

    # Encode family (string to integer)
    le_family = LabelEncoder()

    # Add temporal features (use date column, not index)
    for df in [train_df, val_df]:
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Create lag feature (shift 1 day to avoid data leakage)
    train_df["sale_lag"] = train_df.groupby(["store_nbr", "family"])["sales"].shift(1)
    train_df["sale_lag"] = train_df["sale_lag"].fillna(0)

    # For validation data, create lag using last available training sales
    val_df = val_df.sort_values(["store_nbr", "family", "date"])
    train_last_sales = (
        train_df.groupby(["store_nbr", "family"])["sales"].last().reset_index()
    )
    train_last_sales = train_last_sales.rename(columns={"sales": "sale_lag"})
    val_df = val_df.merge(train_last_sales, on=["store_nbr", "family"], how="left")
    val_df["sale_lag"] = val_df["sale_lag"].fillna(0)

    # Create rolling features (only on training data to avoid leakage)
    # Use min_periods=window to ensure consistent window size (no edge effects)
    for window in [7, 30]:
        train_df[f"rolling_mean_{window}"] = train_df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=window).mean()
        )
        train_df[f"rolling_std_{window}"] = train_df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=window).std()
        )

    # Note: NOT filling NaN in rolling features - will drop warmup rows later
    # This ensures consistent window size and prevents edge effects

    # For validation data, use last rolling values from training
    train_df = train_df.sort_values(["store_nbr", "family", "date"])
    for window in [7, 30]:
        train_last_rolling = (
            train_df.groupby(["store_nbr", "family"])[
                [f"rolling_mean_{window}", f"rolling_std_{window}"]
            ]
            .last()
            .reset_index()
        )
        val_df = val_df.merge(
            train_last_rolling,
            on=["store_nbr", "family"],
            how="left",
            suffixes=("", "_train"),
        )
        val_df[f"rolling_mean_{window}"] = (
            val_df[f"rolling_mean_{window}"]
            .fillna(val_df.get(f"rolling_mean_{window}_train", 0))
            .fillna(0)
        )
        val_df[f"rolling_std_{window}"] = (
            val_df[f"rolling_std_{window}"]
            .fillna(val_df.get(f"rolling_std_{window}_train", 0))
            .fillna(0)
        )

    # Add min/max rolling (7-day window)
    # Use min_periods=7 to ensure consistent window size
    train_df["rolling_min_7"] = train_df.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=7).min())
    train_df["rolling_max_7"] = train_df.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=7).max())

    # Note: NOT filling NaN - will drop warmup rows later

    # For validation data
    train_last_minmax = (
        train_df.groupby(["store_nbr", "family"])[["rolling_min_7", "rolling_max_7"]]
        .last()
        .reset_index()
    )
    val_df = val_df.merge(train_last_minmax, on=["store_nbr", "family"], how="left")
    val_df["rolling_min_7"] = val_df["rolling_min_7"].fillna(0)
    val_df["rolling_max_7"] = val_df["rolling_max_7"].fillna(0)

    # ===== DROP WARMUP ROWS (v4.1c) =====
    # Remove rows where rolling features don't have full window (prevents edge effects)
    # This ensures consistent window size and prevents high variance at series boundaries
    print("  Dropping warmup rows for consistent rolling windows...")

    # Calculate row number per store-family combination
    train_df = train_df.sort_values(["store_nbr", "family", "date"])
    train_df["_row_num"] = train_df.groupby(["store_nbr", "family"]).cumcount()

    # Drop rows where 30-day rolling window isn't complete (max window size)
    rows_before = len(train_df)
    train_df = train_df[train_df["_row_num"] >= 30].copy()
    rows_dropped = rows_before - len(train_df)

    print(
        f"    Dropped {rows_dropped:,} warmup rows ({rows_dropped / rows_before * 100:.2f}% of data)"
    )
    print(f"    Remaining: {len(train_df):,} training rows")

    # Clean up temporary column
    train_df = train_df.drop(columns=["_row_num"])

    # Note: Validation data doesn't need warmup rows dropped because rolling features
    # use the LAST value from training data (static, not recalculated)
    # ===== END WARMUP ROW DROPPING =====

    # ===== LOG TRANSFORM SALES-DEPENDENT FEATURES (v4.1e) =====
    # Lag and rolling features are created from RAW sales, but the target (sales)
    # is log-transformed. This creates a 10x scale mismatch that causes severe
    # overfitting. We must log transform these features to match the target scale.
    print("  Applying log1p transform to sales-dependent features...")
    lag_and_rolling_cols = [
        "sale_lag",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",
        "rolling_std_30",
        "rolling_min_7",
        "rolling_max_7",
    ]

    # Transform training data
    for col in lag_and_rolling_cols:
        if col in train_df.columns:
            before_min = train_df[col].min()
            before_max = train_df[col].max()
            train_df[col] = np.log1p(train_df[col])
            after_min = train_df[col].min()
            after_max = train_df[col].max()
            print(
                f"    {col}: [{before_min:.2f}, {before_max:.2f}] → [{after_min:.4f}, {after_max:.4f}]"
            )

    # Transform validation data
    for col in lag_and_rolling_cols:
        if col in val_df.columns:
            val_df[col] = np.log1p(val_df[col])

    # Verify all features are now on similar scale as target
    print(
        f"    Sales (target) range: [{train_df['sales'].min():.4f}, {train_df['sales'].max():.4f}]"
    )
    print(f"    Features now on log scale ✓")
    # ===== END LOG TRANSFORM SALES-DEPENDENT FEATURES =====

    # Define feature columns (dynamic based on one-hot encoding - v4.1a)
    # Note: store_nbr is NOT a feature - it's used for filtering and as a model parameter

    # Original 14 features
    base_feature_cols = [
        "onpromotion",
        "oil_price",
        "sale_lag",
        "day",
        "month",
        "quarter",
        "year",
        "day_of_week",
        "is_weekend",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",
        "rolling_std_30",
        "rolling_min_7",
        "rolling_max_7",
        # Holiday binary features (not one-hot)
        "is_holiday",
        "is_national_holiday",
        # Store numerical feature
        "cluster",  # store_cluster (already numerical 1-17)
        # Target encoded features
        "city_encoded",
        "state_encoded",
    ]

    # Add one-hot encoded columns dynamically (generated during preprocessing)
    one_hot_cols = [
        c
        for c in train_df.columns
        if c.startswith("holiday_") or c.startswith("store_type_")
    ]

    # Combine all features
    feature_cols = base_feature_cols + one_hot_cols

    print(
        f"  Total features: {len(feature_cols)} (14 base + 2 holiday binary + 1 cluster + 2 target encoded + {len(one_hot_cols)} one-hot)"
    )

    # ===== CRITICAL VALIDATION: Check for NaN/Inf values BEFORE scaling =====
    print("  Validating feature data quality (before scaling)...")

    # Check for NaN values
    train_nan = train_df[feature_cols].isnull().sum()
    val_nan = val_df[feature_cols].isnull().sum()

    if train_nan.any():
        nan_features = train_nan[train_nan > 0]
        print(f"    ❌ ERROR: NaN values detected in training data BEFORE scaling!")
        print(f"       Features with NaN: {nan_features.to_dict()}")
        raise ValueError(
            "NaN values in features before scaling! "
            "Check lag/rolling feature creation for missing shift() operations."
        )

    if val_nan.any():
        nan_features = val_nan[val_nan > 0]
        print(f"    ❌ ERROR: NaN values detected in validation data BEFORE scaling!")
        print(f"       Features with NaN: {nan_features.to_dict()}")
        raise ValueError(
            "NaN values in validation features before scaling! "
            "Check lag/rolling feature creation for missing shift() operations."
        )

    print(f"    ✓ No NaN values in features")

    # Check for Infinite values
    train_inf = (train_df[feature_cols] == np.inf).sum() + (
        train_df[feature_cols] == -np.inf
    ).sum()
    val_inf = (val_df[feature_cols] == np.inf).sum() + (
        val_df[feature_cols] == -np.inf
    ).sum()

    if train_inf.any():
        inf_features = train_inf[train_inf > 0]
        print(
            f"    ❌ ERROR: Infinite values detected in training data BEFORE scaling!"
        )
        print(f"       Features with Inf: {inf_features.to_dict()}")
        raise ValueError(
            "Infinite values in features before scaling! "
            "Check for division by zero in feature engineering."
        )

    if val_inf.any():
        inf_features = val_inf[val_inf > 0]
        print(
            f"    ❌ ERROR: Infinite values detected in validation data BEFORE scaling!"
        )
        print(f"       Features with Inf: {inf_features.to_dict()}")
        raise ValueError(
            "Infinite values in validation features before scaling! "
            "Check for division by zero in feature engineering."
        )

    print(f"    ✓ No infinite values in features")

    # Check for extreme values (potential data quality issues)
    extreme_warnings = 0
    for col in feature_cols:
        train_min = train_df[col].min()
        train_max = train_df[col].max()
        val_min = val_df[col].min()
        val_max = val_df[col].max()

        # Check for suspiciously large values (before scaling)
        if train_max > 1e6 or val_max > 1e6:
            print(f"    ⚠️  WARNING: {col} has extremely large values")
            print(f"       Train range: [{train_min:.2f}, {train_max:.2f}]")
            print(f"       Val range: [{val_min:.2f}, {val_max:.2f}]")
            print(f"       This might indicate a data quality issue")
            extreme_warnings += 1

    if extreme_warnings == 0:
        print(f"    ✓ No extreme value warnings")

    print(f"    ✓ Feature data quality validation complete\n")

    # Scale features (fit on training data only!)
    print("  Scaling features...")
    scaler_X = StandardScaler()

    # Store raw means BEFORE scaling for distribution analysis
    raw_train_means = train_df[feature_cols].mean()
    raw_val_means = val_df[feature_cols].mean()

    # Fit scaler on training data
    train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler_X.transform(val_df[feature_cols])

    # ===== VALIDATION: Distribution Analysis (RAW data) =====
    print("  Analyzing raw feature distributions...")

    # 1. Show scaler learned parameters (what the scaler "knows" about the data)
    print(f"    Train feature means (first 5):")
    for i, col in enumerate(feature_cols[:5]):
        print(f"{col}: {scaler_X.mean_[i]:.4f}")

    # 2. Check for distribution shift (raw data BEFORE scaling)
    mean_diff_raw = (raw_val_means - raw_train_means).abs()
    max_mean_diff = mean_diff_raw.max()
    max_mean_diff_feature = mean_diff_raw.idxmax()

    print(f"\n    Val vs Train mean difference (raw, BEFORE scaling):")
    print(
        f"      Max difference: {max_mean_diff:.4f} (feature: {max_mean_diff_feature})"
    )
    print(f"      Mean difference: {mean_diff_raw.mean():.4f}")

    # Check if any feature has large distribution shift (> 50% of train mean)
    relative_shift = mean_diff_raw / (raw_train_means.abs() + 1e-8)
    large_shifts = relative_shift[relative_shift > 0.5]

    if len(large_shifts) > 0:
        print(f"\n    ⚠️  WARNING: {len(large_shifts)} features with >50% mean shift!")
        print(f"       These features have different distributions in train vs val:")
        for col in large_shifts.head(3).index:
            train_mean = raw_train_means[col]
            val_mean = raw_val_means[col]
            shift_pct = relative_shift[col] * 100
            print(
                f"{col}: train={train_mean:.2f}, val={val_mean:.2f} ({shift_pct:.1f}% shift)"
            )
    else:
        print(f"    ✓ No large distribution shifts detected")

    # 3. Check for features with very different scales (before scaling)
    print(f"\n    Feature ranges (raw, BEFORE scaling):")
    raw_ranges = pd.DataFrame(
        {
            "min": train_df[feature_cols].min(),  # These are actually scaled now
            "max": train_df[feature_cols].max(),  # Need to use raw values
        }
    )
    # Oops, I need to use the raw values before scaling
    print(f"      (Validation uses scaled data to check for issues)\n")

    # ===== VALIDATION: Check for scaling issues =====
    print("  Validating scaling...")

    # 1. Check for extreme values (potential outliers)
    train_scaled = train_df[feature_cols].values
    val_scaled = val_df[feature_cols].values

    # Check if any values are extreme (> 10 standard deviations)
    extreme_threshold = 10
    train_extreme = np.abs(train_scaled) > extreme_threshold
    val_extreme = np.abs(val_scaled) > extreme_threshold

    if train_extreme.any():
        extreme_count = train_extreme.sum()
        extreme_features = [
            col
            for col in feature_cols
            if train_scaled[:, feature_cols.index(col)].max() > extreme_threshold
        ]
        print(
            f"    ⚠️  WARNING: {extreme_count} scaled values > {extreme_threshold}σ in training data"
        )
        print(f"       Features with extremes: {extreme_features[:5]}")
    else:
        print(f"    ✓ No extreme values (> {extreme_threshold}σ) in training data")

    # 2. Check for distribution shift between train and val (using mean/std)
    train_means = train_df[feature_cols].mean()
    val_means = val_df[feature_cols].mean()

    # Calculate mean shift (should be close to 0 after proper scaling)
    mean_shifts = np.abs(val_means - train_means)
    max_shift = mean_shifts.max()

    if max_shift > 2.0:
        shifted_features = mean_shifts[mean_shifts > 2.0].index.tolist()
        print(
            f"    ⚠️  WARNING: Mean shift > 2σ detected for {len(shifted_features)} features"
        )
        print(f"       Max shift: {max_shift:.2f}σ")
        print(f"       Shifted features: {shifted_features[:5]}")
    else:
        print(f"    ✓ No significant distribution shift (max shift: {max_shift:.2f}σ)")

    # 3. Check for missing values after scaling (should be 0)
    train_missing = train_df[feature_cols].isnull().sum().sum()
    val_missing = val_df[feature_cols].isnull().sum().sum()

    if train_missing > 0 or val_missing > 0:
        print(f"    ❌ ERROR: Missing values detected after scaling!")
        print(f"       Train: {train_missing}, Val: {val_missing}")
        raise ValueError("Missing values after scaling - check preprocessing pipeline")
    else:
        print(f"    ✓ No missing values after scaling")

    # 4. Summary statistics
    print(f"    Train range: [{train_scaled.min():.2f}, {train_scaled.max():.2f}]")
    print(f"    Val range: [{val_scaled.min():.2f}, {val_scaled.max():.2f}]")
    print(f"    Train mean (should be ~0): {train_scaled.mean():.4f}")
    print(f"    Val mean (should be ~0): {val_scaled.mean():.4f}")

    # Transform target (sales) using log1p for RMSLE loss
    # Kaggle RMSLE metric: sqrt(mean((log(pred + 1) - log(true + 1))^2))
    train_df["sales"] = np.log1p(train_df["sales"])
    val_df["sales"] = np.log1p(val_df["sales"])

    print(f"  ✓ Created {len(feature_cols)} features")

    # Prepare preprocessing metadata for inference
    # This ensures inference uses EXACTLY the same preprocessing as training
    preprocessing_metadata = {
        "city_mean_sales": city_mean_sales,  # Target encoding means
        "state_mean_sales": state_mean_sales,  # Target encoding means
        "one_hot_cols": one_hot_cols,  # One-hot column names
        "feature_cols": feature_cols,  # All feature column names (in order)
        "base_feature_cols": base_feature_cols,  # Base features (before one-hot)
        "feature_count": len(feature_cols),  # Total feature count
    }

    return train_df, val_df, scaler_X, preprocessing_metadata


class EarlyStopping:
    """Early stops training when validation loss doesn't improve."""

    def __init__(
        self, patience: int = 5, min_delta: float = 0.001, verbose: bool = True
    ):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  ✅ Initial best loss: {val_loss:.6f}")
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  ✅ New best loss: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏸ No improvement ({self.counter}/{self.patience})")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"  🛑 Early stopping triggered!")

        return self.early_stop


def save_best_model(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    best_val_loss,
    path,
    num_features=None,
    num_families=None,
    num_stores=None,
    feature_cols=None,
    window_size=None,
):
    """
    Save best model checkpoint with dimension metadata.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        best_val_loss: Best validation loss achieved
        path: Path to save checkpoint
        num_features: Number of input features (required)
        num_families: Number of product families (required)
        num_stores: Number of stores (required)
        feature_cols: List of feature column names (required for v4.1a)
    """
    # Build checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "model_type": "multitask_lstm",
    }

    # Add dimension metadata (required for inference)
    if num_features is not None:
        checkpoint["num_features"] = num_features
    if num_families is not None:
        checkpoint["num_families"] = num_families
    if num_stores is not None:
        checkpoint["num_stores"] = num_stores

    # Add feature column names (required for v4.1a inference validation)
    if feature_cols is not None:
        checkpoint["feature_cols"] = feature_cols

    # Add window size (required for inference to match training context)
    if window_size is not None:
        checkpoint["window_size"] = window_size

    # Validate all dimensions are present
    required_metadata = ["num_features", "num_families", "num_stores"]
    missing = [key for key in required_metadata if key not in checkpoint]

    if missing:
        raise ValueError(
            f"Cannot save checkpoint: missing required metadata {missing}\n"
            f"Please provide: num_features, num_families, num_stores"
        )

    # Warn if feature_cols not provided (v4.1a requirement)
    if "feature_cols" not in checkpoint:
        print(f"⚠️  WARNING: feature_cols not saved in checkpoint!")
        print(f"   Inference will not be able to validate feature names.")

    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"✓ Saved best model to: {path}")


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu", model_type="lstm"):
    """
    Load checkpoint and restore model/optimizer state.

    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        start_epoch: Epoch to resume from
        train_losses: Training history
        val_losses: Validation history
        best_val_loss: Best validation loss
        scaler_X: Loaded scaler (if in checkpoint)
        scaler_y: Loaded scaler (if in checkpoint)
    """
    # Load checkpoint (weights_only=False needed for sklearn scalers)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if it's a full checkpoint or just state_dict
    if "model_state_dict" in checkpoint:
        # Full checkpoint with config

        # Display version info if available
        if "model_version" in checkpoint:
            print(f"\n{'=' * 70}")
            print("LOADED CHECKPOINT INFO")
            print(f"{'=' * 70}")
            print(f"Version:       {checkpoint.get('model_version', 'UNKNOWN')}")
            print(f"Description:   {checkpoint.get('model_description', 'N/A')}")
            print(f"Model Type:    {checkpoint.get('model_type', model_type)}")
            print(f"Trained Date:  {checkpoint.get('trained_date', 'N/A')}")
            print(f"Num Features:  {checkpoint.get('num_features', 'N/A')}")
            print(f"{'=' * 70}\n")

        model_config = checkpoint.get("model_config", {})

        # Recreate model with saved config if available
        if model_config:
            print(f"Resuming with config: {model_config}")

        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✓ Optimizer state restored")

        # Load training history
        start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        best_val_loss = checkpoint.get(
            "best_val_loss", min(val_losses) if val_losses else float("inf")
        )

        # Load scalers if available
        scaler_X = checkpoint.get("scaler_X", None)
        scaler_y = checkpoint.get("scaler_y", None)

        if scaler_X is not None and scaler_y is not None:
            print("✓ Scalers loaded from checkpoint")

        print(f"✓ Resuming from epoch {start_epoch}")
        print(
            f"✓ Previous train losses: {train_losses[-3:] if train_losses else 'N/A'}"
        )
        print(f"✓ Previous val losses: {val_losses[-3:] if val_losses else 'N/A'}")
        print(f"✓ Best val loss so far: {best_val_loss:.6f}")

        return (
            model,
            optimizer,
            start_epoch,
            train_losses,
            val_losses,
            best_val_loss,
            scaler_X,
            scaler_y,
        )
    else:
        # Just state_dict
        model.load_state_dict(checkpoint)
        print("✓ Model weights loaded")
        print("⚠ No training history found, starting from epoch 0")

        return (
            model,
            optimizer,
            0,
            [],
            [],
            float("inf"),
            None,
            None,
        )
