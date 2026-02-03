"""
Meta-Learning Multi-Task Data Loader

Handles family-level feature granularity by creating 3D tensors:
[Batch, Dates, Features, Families]

This allows the model to predict all 33 families for a store simultaneously.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import joblib
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to DataFrame.

    Args:
        df: DataFrame with 'date' column or date index

    Returns:
        DataFrame with added temporal features
    """
    # Ensure date is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Set date as index if not already
    if df.index.name != "date" and "date" in df.columns:
        df = df.set_index("date")

    # Add temporal features
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


class MetaLearningDataLoader(Dataset):
    """
    Store-centric data loader for meta-learning multi-task forecasting.

    For each store:
    - Input:  [Window-1 days, 15 features, 33 families] - 3D tensor
    - Output: [33] - sales for all families on prediction day
    """

    def __init__(
        self,
        df: pd.DataFrame,
        store_nbr: int,
        window_size: int = 14,  # Reduced from 30 for faster training
        log_transform: bool = True,
    ):
        """
        Args:
            df: Processed DataFrame with all features
            store_nbr: Store number to load data for
            window_size: Number of days in input sequence
            log_transform: Whether to use log1p transform on sales
        """
        self.store_nbr = store_nbr
        self.window_size = window_size
        self.log_transform = log_transform

        # NOTE: log_transform is DISABLED for v4.0+
        # Target (sales) is already log1p-transformed in training.py (line 301-302)
        # Applying log1p again would cause DOUBLE TRANSFORM BUG
        if self.log_transform:
            raise ValueError(
                "log_transform=True is NOT supported in v4.0+! "
                "Sales are already log-transformed in preprocessing. "
                "Set log_transform=False or remove this parameter."
            )

        # Filter data for this store
        store_df = df[df["store_nbr"] == store_nbr].copy()

        # Get metadata columns
        if "date" in store_df.columns:
            store_df["date"] = pd.to_datetime(store_df["date"])
            store_df = store_df.set_index("date")

        # Define column types
        exclude_cols = ["store_nbr", "family", "sales", "item_enc", "index", "id"]
        self.feature_cols = [col for col in store_df.columns if col not in exclude_cols]

        # Create pivoted features: [dates x families] for each feature
        self.feature_pivots = {}
        for col in self.feature_cols:
            pivot = store_df.pivot_table(
                index=store_df.index, values=col, columns="family", aggfunc="mean"
            )
            self.feature_pivots[col] = pivot

        # Get sales target: [dates x families]
        self.sales = store_df.pivot_table(
            index=store_df.index, values="sales", columns="family", aggfunc="mean"
        )

        # Get dates
        self.dates = self.sales.index.unique()

        # Create sequences
        self.num_samples = len(self.dates) - window_size + 1
        print(f"  Created {self.num_samples} sequences from {len(self.dates)} dates")

        # PRE-COMPUTE ALL TENSORS for O(1) __getitem__ access
        # This is a 10-100× speedup vs. constructing tensors on-the-fly
        print(f"  Pre-computing {self.num_samples} 3D tensors...")
        self.X_all = []
        self.y_all = []

        for idx in range(self.num_samples):
            end_idx = idx + self.window_size - 1

            # Build 3D tensor: [window-1 dates, features, families]
            feature_slices = []
            for date_idx in range(idx, idx + self.window_size - 1):
                date = self.dates[date_idx]

                # Get all features for this date across all families
                feature_matrix = np.stack(
                    [
                        self.feature_pivots[col].loc[date].values
                        for col in self.feature_cols
                    ],
                    axis=0,
                )  # [features, families]
                feature_slices.append(feature_matrix)

            # Stack along time dimension: [window-1, features, families]
            X = np.stack(feature_slices, axis=0)
            y = self.sales.iloc[end_idx].values  # [families]

            self.X_all.append(torch.tensor(X, dtype=torch.float32))
            self.y_all.append(torch.tensor(y, dtype=torch.float32))

        print(f"  Pre-computation complete: {len(self.X_all)} samples ready")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) tensor lookup - all tensors pre-computed in __init__

        Returns:
            X: [Window-1, Features, Families] - 3D tensor
            y: [Families] - target sales for next day
        """
        return self.X_all[idx], self.y_all[idx]


def create_store_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    store_to_test: int,
    batch_size: int,
    window_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for a single store (POC).

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        store_to_test: Which store to test (1-54)
        batch_size: Batch size
        window_size: Sequence window size
        num_workers: Number of workers for DataLoader

    Returns:
        train_loader, val_loader
    """
    print(f"\n{'=' * 60}")
    print(f"Creating DataLoaders for Store {store_to_test} (POC)")
    print(f"{'=' * 60}")

    # Create datasets
    train_ds = MetaLearningDataLoader(
        train_df, store_nbr=store_to_test, window_size=window_size, log_transform=False
    )

    val_ds = MetaLearningDataLoader(
        val_df, store_nbr=store_to_test, window_size=window_size, log_transform=False
    )

    print(f"\n✓ Created datasets:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle time series!
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    # Test one batch
    print(f"\n🔍 Testing one batch...")
    X_batch, y_batch = next(iter(train_loader))
    print(f"  Input shape:  {X_batch.shape} (Batch, Dates, Features, Families)")
    print(f"  Output shape: {y_batch.shape} (Batch, Families)")
    print(f"  Input range:  [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print(f"  Output range: [{y_batch.min():.3f}, {y_batch.max():.3f}]")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Load processed data with error handling
    logger.info("Loading processed data...")

    try:
        train_df = pd.read_csv("./data/train_processed.csv")
        logger.info(f"✓ Loaded {len(train_df):,} training rows")
    except FileNotFoundError:
        raise FileNotFoundError(
            "❌ train_processed.csv not found in ./data/\n"
            "Run create_processed_data.py first."
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load training data: {str(e)}")

    try:
        val_df = pd.read_csv("./data/val_processed.csv")
        logger.info(f"✓ Loaded {len(val_df):,} validation rows")
    except FileNotFoundError:
        raise FileNotFoundError(
            "❌ val_processed.csv not found in ./data/\n"
            "Run create_processed_data.py first."
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load validation data: {str(e)}")

    # Validate data
    if train_df.empty:
        raise ValueError("❌ Training data is empty!")
    if val_df.empty:
        raise ValueError("❌ Validation data is empty!")

    # Test with Store 1
    train_loader, val_loader = create_store_dataloaders(
        train_df, val_df, store_to_test=1, batch_size=32, window_size=14
    )

    logger.info("")
    logger.info("✅ Data loader test passed!")
