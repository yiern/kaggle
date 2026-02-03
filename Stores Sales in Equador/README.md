# Store Sales Time Series Forecasting

**Kaggle Competition**: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Predict daily sales for 54 Favorita stores across Ecuador using a meta-learning multi-task LSTM architecture.

---

## 🎯 Strategy Overview

### Core Approach: Meta-Learning Multi-Task LSTM

This project implements a **hierarchical meta-learning architecture** that leverages:

1. **Shared Knowledge** - A global LSTM learns temporal patterns common across all 54 stores
2. **Store Adaptation** - Store-specific adapters (54 modules) learn local adjustments
3. **Multi-Task Learning** - Predicts all 33 product families simultaneously
4. **3D Tensor Representation** - `[Batch, Window, Features, Families]` for efficient family-wise feature modeling

### Why This Works

| Problem | Solution |
|---------|----------|
| **Store heterogeneity** | Store-specific adapters capture local patterns |
| **Feature scale mismatch** | Feature encoder (32→16 dims) acts as regularization |
| **Family interdependence** | Multi-task output learns family correlations |
| **Temporal leakage** | Strict chronological splits (no random shuffling) |
| **Data leakage** | Scalers fit on training only, lag features use `shift(1)` |

---

## 🏗️ Model Architecture (v4.1f)

```
Input: [Batch, 6, 32, 33]  ← Window=7, 32 features, 33 families
        │
        ▼
┌─────────────────────────────────────────────┐
│  Feature Encoder (32 → 16 dims per family)  │
│  Linear → ReLU → Dropout (acts as reg.)      │
└─────────────────────────────────────────────┘
        │ Output: [Batch, 6, 528]  (16 × 33)
        ▼
┌─────────────────────────────────────────────┐
│  Shared LSTM Encoder                         │
│  Input: 528 → Hidden: 256 × 2 layers        │
│  Learns cross-store temporal patterns        │
└─────────────────────────────────────────────┘
        │ Output: [Batch, 256]
        ▼
┌─────────────────────────────────────────────┐
│  Store Adapter (store_nbr specific)         │
│  Linear → LayerNorm → ReLU → Dropout         │
│  54 separate adapters (one per store)        │
└─────────────────────────────────────────────┘
        │ Output: [Batch, 256]
        ▼
┌─────────────────────────────────────────────┐
│  Family Predictor                            │
│  256 → 128 → 33 (all families)              │
└─────────────────────────────────────────────┘
        │
        ▼
Output: [Batch, 33]  ← Log-transformed sales predictions
```

### Parameters

| Component | Parameters |
|-----------|------------|
| Shared LSTM | ~1.6M |
| Store Adapters | ~3.5M |
| Family Predictor | ~34K |
| **Total** | **~5.1M** |

---

## 📅 Version History & Timeline

| Version | Date | Changes | Impact |
|---------|------|---------|--------|
| **v4.1f** | 2026-02-03 | Pre-computed 3D tensors in DataLoader (O(1) lookups) | 10-100× faster data loading |
| **v4.1e** | 2026-02-02 | Fixed feature-target scale mismatch with `log1p()` on sales-dependent features | Better alignment between features and target |
| **v4.1c** | 2026-02-02 | Fixed rolling feature edge effects (`min_periods=window`) | Eliminated NaN spikes at series boundaries |
| **v4.1b** | 2026-02-02 | Window size optimization: `window_size=7` (was 14) | **6.08% better**, 4× faster training |
| **v4.0** | 2026-02-01 | Meta-Learning Multi-Task LSTM (production release) | Baseline architecture |

---

## 🛠️ Feature Engineering Pipeline

### 1. External Data Merging
- Oil prices (daily)
- Holidays/events (national, regional, local)
- Transactions (per store)
- Store metadata (cluster, city, state, type)

### 2. Temporal Features
```python
day, month, quarter, year, day_of_week, is_weekend
```

### 3. Lag & Rolling Features (Data Leakage Safe!)
```python
# CRITICAL: Always use shift(1) for PAST data only
sale_lag = df.groupby(['store_nbr', 'family'])['sales'].shift(1)

rolling_mean_7 = df.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
)
```

### 4. Holiday Features
- `is_holiday`, `is_national_holiday`
- 6 one-hot encoded holiday types

### 5. Store Features
- `cluster`, `city_encoded`, `state_encoded`
- 5 one-hot encoded store types

### 6. Scaling & Encoding
- **Numerical**: `StandardScaler` (fit on train only!)
- **Categorical**: `LabelEncoder` for city, state
- **Target**: `log1p(sales)` for RMSLE optimization

### Feature Count: 32 numerical features

---

## 🚀 Training Strategy

### Mixed Batch Training

```python
for epoch in range(NUM_EPOCHS):
    random.shuffle(stores_list)  # Different order each epoch
    for store_id in stores_list:
        train_loader, _ = store_loaders[store_id]
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch, store_nbr=store_id)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            break  # One batch per store per epoch
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.00025 |
| Weight Decay | 0.05 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch Size | 128 |
| Window Size | 7 (6 timesteps in, 1 prediction) |
| Early Stopping | Patience=5, min_delta=0.001 |
| Gradient Clipping | max_norm=1.0 |

### Loss Function: RMSLE

```python
RMSLE = sqrt(MSE(log(pred + 1), log(actual + 1)))
```

Optimized directly on log-transformed sales.

---

## 📂 Project Structure

```
.
├── data/
│   ├── train_processed.csv          # Preprocessed training data
│   ├── val_processed.csv            # Preprocessed validation data
│   ├── scaler_X.pkl                 # Feature scaler (for inference)
│   └── preprocessing_metadata.pkl   # Feature names & encoding info
├── models/
│   └── best_model_multitask.pt      # Saved model checkpoint
├── ML_Models/
│   └── multitask_lstm.py            # Model architecture (v4.1f)
├── src/
│   ├── data_multitask.py            # 3D tensor DataLoader
│   ├── training.py                  # Training utilities
│   └── new_store_utils.py           # Store-specific helpers
├── utils/
│   └── logger.py                    # Logging configuration
├── train_model.py                   # Training script
├── inference.py                     # Kaggle submission generator
└── create_processed_data.py         # Feature engineering pipeline
```

---

## 🏃 Quick Start

### 1. Create Processed Data
```bash
python create_processed_data.py
```
Output: `data/train_processed.csv`, `data/val_processed.csv`

### 2. Train Model
```bash
python train_model.py
```
Output: `models/best_model_multitask.pt`

### 3. Generate Submission
```bash
python inference.py
```
Output: `submission.csv`

---

## ⚙️ Configuration

### Device: Apple Silicon (MPS)
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Training/Validation Split
```python
split_date = '2017-07-15'  # Chronological split (no shuffling!)
train = df[df.index < split_date]
val = df[df.index >= split_date]
```

---

## 🎓 Key Learnings

### What Worked

1. **Store adapters** - Critical for handling store heterogeneity
2. **Window size 7** - Optimal balance (smaller was worse, larger was slower)
3. **Feature compression** - 32→16 dims prevented overfitting
4. **Pre-computed tensors** - Massive data loading speedup
5. **Log-transform on sales-dependent features** - Fixed feature-target scale mismatch

### What Didn't Work

1. **Larger windows** (14, 30) - More noise, slower, worse performance
2. **No store adapters** - Stores too different for shared model only
3. **Random shuffling** - Temporal leakage destroyed generalization
4. **Fitting scalers on full data** - Data leakage inflated validation scores

---

## 📊 Performance

### Training Speed (v4.1f)
- **Data loading**: 10-100× faster (pre-computed tensors)
- **Epoch time**: ~20-30 seconds (54 stores, mixed batch)

### Model Metrics
- **Parameters**: ~5.1M trainable
- **Input**: 6 timesteps × 32 features × 33 families
- **Output**: 33 family sales predictions

---

## 🔧 Development Commands

```bash
# Linting (critical errors only)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Format code
black . --line-length 100

# Train from scratch
python train_model.py

# Resume from checkpoint (auto-detects)
python train_model.py
```

---

## 📝 License

This project is part of a Kaggle competition. See [Kaggle Terms](https://www.kaggle.com/terms) for competition-specific rules.

---

## 🤝 Contributing

This is a solo competition project. For questions or issues, please open an issue on GitHub.
