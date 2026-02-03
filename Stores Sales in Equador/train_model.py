"""
Training script for Store Sales Time Series Forecasting (v4.1f Performance Optimizations).
Uses mixed batch training with 3D hierarchical tensors.
Saves best model automatically based on validation loss.

Changes in v4.1f:
- Pre-computed 3D tensors in DataLoader (10-100× faster data loading)
- Fixed MPS device detection for Apple Silicon inference
"""

import random
import time
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ML_Models.multitask_lstm import MetaLearningMultiTaskLSTM, RMSLELoss
from src.data_multitask import create_store_dataloaders
from src.training import EarlyStopping, save_best_model
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, log_file="logs/training.log")


def create_store_dataloaders_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    window_size: int,
) -> dict:
    """
    Create dataloaders for all 54 stores.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        batch_size: Batch size for each store
        window_size: Time series window size

    Returns:
        Dictionary with store_id as key, (train_loader, val_loader) as value
    """
    store_loaders = {}

    for store_id in range(1, 55):  # Stores 1-54
        try:
            train_loader, val_loader = create_store_dataloaders(
                train_df=train_df,
                val_df=val_df,
                store_to_test=store_id,
                batch_size=batch_size,
                window_size=window_size,
                num_workers=0,  # Mac optimization
            )
            store_loaders[store_id] = (train_loader, val_loader)
        except Exception as e:
            print(f"  Warning: Failed to create dataloader for store {store_id}: {e}")
            continue

    return store_loaders


def main():
    logger.info("=" * 70)
    logger.info("STORE SALES FORECASTING - META-LEARNING TRAINING (v4.0)")
    logger.info("=" * 70)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load processed data with error handling
    logger.info("")
    logger.info("Loading processed data...")
    train_path = Path("./data/train_processed.csv")
    val_path = Path("./data/val_processed.csv")

    if not train_path.exists() or not val_path.exists():
        logger.error("❌ Processed data not found!")
        logger.error("   Run: python create_processed_data.py")
        sys.exit(1)

    try:
        train_df = pd.read_csv(train_path)
        logger.info(f"✓ Loaded {len(train_df):,} training rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ train_processed.csv not found at {train_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load training data: {str(e)}")

    try:
        val_df = pd.read_csv(val_path)
        logger.info(f"✓ Loaded {len(val_df):,} validation rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ val_processed.csv not found at {val_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load validation data: {str(e)}")

    # Validate data is not empty
    if train_df.empty:
        raise ValueError("❌ Training data is empty!")
    if val_df.empty:
        raise ValueError("❌ Validation data is empty!")

    # Ensure date column exists and parse it
    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"])
        val_df["date"] = pd.to_datetime(val_df["date"])

    # Create dataloaders for all stores
    logger.info("Creating dataloaders for all 54 stores...")
    batch_size = 128
    window_size = 7

    store_loaders = create_store_dataloaders_all(
        train_df, val_df, batch_size=batch_size, window_size=window_size
    )

    logger.info(f"✓ Created dataloaders for {len(store_loaders)} stores")

    if len(store_loaders) == 0:
        logger.error("❌ No store dataloaders created!")
        sys.exit(1)

    # Initialize model
    # Derive dimensions dynamically from data
    exclude_cols = {"sales", "family", "item_enc", "date", "index", "id", "store_nbr"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    num_features = len(feature_cols)

    # Derive num_families and num_stores from data
    num_families = int(train_df["family"].nunique())
    num_stores = int(train_df["store_nbr"].nunique())

    logger.info("")
    logger.info("Model configuration (derived from data):")
    logger.info(f"  Features:  {num_features}")
    logger.info(f"  Families:  {num_families}")
    logger.info(f"  Stores:    {num_stores}")
    logger.info(f"  Batch size:   {batch_size}")
    logger.info(f"  Window size:  {window_size}")

    model = MetaLearningMultiTaskLSTM(
        num_features=num_features,
        num_families=num_families,
        num_stores=num_stores,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
    ).to(device)

    # Training setup
    criterion = RMSLELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Early stopping setup
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)



    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")

    # ===== MODEL ARCHITECTURE BREAKDOWN =====
    print(f"\n{'=' * 70}")
    print("MODEL ARCHITECTURE BREAKDOWN (v4.1d)")
    print(f"{'=' * 70}")

    # Count parameters per component
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    store_adapters_params = sum(p.numel() for p in model.store_adapters.parameters())
    family_predictor_params = sum(
        p.numel() for p in model.family_predictor.parameters()
    )

    print(
        f"Shared LSTM (input: {model.num_features * model.num_families} → 256):    {lstm_params:,} params"
    )
    print(
        f"Store Adapters (54 adapters):                        {store_adapters_params:,} params"
    )
    print(
        f"Family Predictor (256 → 33):                         {family_predictor_params:,} params"
    )
    print(f"{'=' * 70}")
    print(f"Total:                                               {num_params:,} params")
    print(f"{'=' * 70}\n")

    print(f"Using RMSLE loss (Kaggle metric)")
    print(f"Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

    # ===== FRESH TRAINING - NO RESUME =====
    best_model_path = "models/best_model_multitask.pt"
    start_epoch = 0
    train_losses = []
    val_losses = []
    gap_history = []  # Track train/val gap for overfitting detection
    best_val_loss = float("inf")

    print(f"Starting fresh training (no checkpoint resume)")
    print(f"Best model will be saved to: {best_model_path}\n")

    # ===== TRAINING LOOP =====
    NUM_EPOCHS = 100
    stores_list = list(store_loaders.keys())
    epoch = None  # Initialize to handle case where loop doesn't run

    # Timing setup
    training_start_time = time.time()
    epoch_times = []

    print("Starting mixed batch training...")
    print(f"Training on {len(stores_list)} stores with random order each epoch")

    # Initialize avg_val_loss BEFORE try block for KeyboardInterrupt safety
    avg_val_loss = float("inf")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()

            # === TRAINING ===
            model.train()
            epoch_train_loss = 0.0
            train_batch_count = 0

            # Randomize store order for this epoch
            random.shuffle(stores_list)

            for store_id in stores_list:
                train_loader, _ = store_loaders[store_id]

                # Train on all batches from this store
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(X_batch, store_nbr=store_id)
                    loss = criterion(predictions, y_batch)

                    # Backward pass
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )
                        optimizer.step()
                        epoch_train_loss += loss.item()
                        train_batch_count += 1

            avg_train_loss = (
                epoch_train_loss / train_batch_count if train_batch_count > 0 else 0.0
            )
            train_losses.append(avg_train_loss)

            # === VALIDATION ===
            model.eval()
            epoch_val_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                # Validate on all stores
                for store_id in stores_list:
                    _, val_loader = store_loaders[store_id]

                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)

                        # Validation
                        predictions = model(X_batch, store_nbr=store_id)
                        loss = criterion(predictions, y_batch)

                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            epoch_val_loss += loss.item()
                            val_batch_count += 1

            avg_val_loss = (
                epoch_val_loss / val_batch_count if val_batch_count > 0 else 0.0
            )
            val_losses.append(avg_val_loss)

            # Calculate and track train/val gap (for overfitting detection)
            gap = avg_val_loss - avg_train_loss
            gap_history.append(gap)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Calculate epoch time and loss ratio
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            loss_ratio = (
                avg_val_loss / avg_train_loss if avg_train_loss > 0 else float("inf")
            )

            # Epoch summary with timing, loss ratio, and train/val gap
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
                f"train_loss: {avg_train_loss:.6f}, val_loss: {avg_val_loss:.6f}, "
                f"gap: {gap:+.6f}, ratio: {loss_ratio:.3f}, time: {epoch_time:.1f}s"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_best_model(
                    model,
                    optimizer,
                    epoch,
                    train_losses,
                    val_losses,
                    best_val_loss,
                    best_model_path,
                    num_features=num_features,
                    num_families=num_families,
                    num_stores=num_stores,
                    feature_cols=feature_cols,
                    window_size=window_size,
                )
                print(f"  ↳ New best model saved! val_loss: {best_val_loss:.6f}")

            # Early stopping
            if early_stopping(avg_val_loss):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best val loss: {best_val_loss:.6f}")
                break

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("⚠️  KEYBOARD INTERRUPT DETECTED")
        print("=" * 70)
        if epoch is not None:
            print(f"Finishing epoch {epoch + 1}...")
        else:
            print("Training was interrupted before starting...")

        # Check if avg_val_loss was computed
        if avg_val_loss != float("inf"):
            print(f"Current validation loss: {avg_val_loss:.6f}")
            print(f"Best validation loss: {best_val_loss:.6f}")

            # Save if this epoch improved validation loss
            if avg_val_loss < best_val_loss:
                print(
                    f"✅ Saving best model (val_loss improved: {best_val_loss:.6f} → {avg_val_loss:.6f})"
                )
                save_best_model(
                    model,
                    optimizer,
                    epoch,
                    train_losses,
                    val_losses,
                    avg_val_loss,
                    best_model_path,
                    num_features=num_features,
                    num_families=num_families,
                    num_stores=num_stores,
                    window_size=window_size,
                    feature_cols=feature_cols,
                )
            else:
                print(f"ℹ️  Not saving (val_loss did not improve)")
        else:
            print(f"ℹ️  Training interrupted before validation completed")
            print(f"Best validation loss so far: {best_val_loss:.6f}")

        print("\n👋 Training stopped by user. Exiting gracefully...")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    final_epoch = epoch + 1 if epoch is not None else start_epoch
    print(f"Total epochs: {final_epoch}")

    # Timing summary
    total_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    print(f"\n{'=' * 70}")
    print("TIMING STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total training time:     {total_time / 60:.1f} minutes")
    print(f"Average time per epoch:  {avg_epoch_time:.1f} seconds")
    print(f"Fastest epoch:           {min(epoch_times):.1f}s" if epoch_times else "")
    print(f"Slowest epoch:           {max(epoch_times):.1f}s" if epoch_times else "")
    print(f"{'=' * 70}\n")

    # Loss summary
    print(f"Best val loss: {best_val_loss:.6f}")
    if train_losses:
        print(f"Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.6f}")

    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
