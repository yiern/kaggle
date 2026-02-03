"""
Meta-Learning Multi-Task LSTM Model v4.1f

Architecture:
1. Feature encoder: Compress features for each family (32 → 16 dims)
2. Shared LSTM: Learns temporal patterns common to all stores
3. Store adapters: Learns store-specific adjustments
4. Family decoder: Predicts all 33 families simultaneously

Changes in v4.1f:
- Optimized DataLoader with pre-computed tensors (O(1) lookups)
- Improved data loading performance by 10-100×

Note: Feature compression (32→16) acts as regularization, preventing overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MetaLearningMultiTaskLSTM(nn.Module):
    """
    Meta-learning multi-task LSTM for hierarchical sales forecasting.

    Predicts sales for all 33 families for a single store using:
    - Shared knowledge across stores (meta-learning)
    - Store-specific adaptations (adapters)
    - Multi-task output (all 33 families)
    """

    def __init__(
        self,
        num_features: int = 32,
        num_families: int = 33,
        num_stores: int = 54,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.25,
        family_embed_dim: int = 16,
    ):
        """
        Args:
            num_features: Number of input features per family (32)
            num_families: Number of product families (33)
            num_stores: Total number of stores (54)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            family_embed_dim: Dimension for feature encoding (acts as regularization)
        """
        super().__init__()

        self.num_features = num_features
        self.num_families = num_families
        self.num_stores = num_stores
        self.hidden_dim = hidden_dim

        # Step 1: Feature encoder (compresses 32 features → 16 dims)
        # Acts as regularization, preventing overfitting
        self.family_encoder = nn.Sequential(
            nn.Linear(num_features, family_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Step 2: Shared LSTM encoder
        # Input: [16 dims × 33 families = 528] dims per timestep
        # Output: [256] hidden representation
        self.lstm = nn.LSTM(
            input_size=family_embed_dim * num_families,  # 16 * 33 = 528
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Step 3: Store-specific adapters
        # One adapter per store to learn local patterns
        self.store_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                )
                for _ in range(num_stores)
            ]
        )

        # Step 4: Family predictor (shared across all stores)
        # Predict all 33 families from hidden state
        self.family_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_families),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, store_nbr: int) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [Batch, Dates, Features, Families]
            store_nbr: Store number (1-54, will be converted to 0-indexed)

        Returns:
            predictions: [Batch, Families] - sales for all families
        """
        # Input validation
        if not 1 <= store_nbr <= self.num_stores:
            raise ValueError(
                f"store_nbr must be in range [1, {self.num_stores}], got {store_nbr}"
            )

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor [Batch, Dates, Features, Families], got {x.ndim}D tensor"
            )

        batch_size, num_dates, num_features, num_families = x.shape

        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}"
            )

        if num_families != self.num_families:
            raise ValueError(
                f"Expected {self.num_families} families, got {num_families}"
            )

        # Step 1: Encode features for each family
        # Reshape: [Batch, Dates, Features, Families] → [Batch*Dates*Families, Features]
        x = x.permute(0, 1, 3, 2)  # [Batch, Dates, Families, Features]
        x = x.reshape(batch_size * num_dates * num_families, num_features)

        # Encode: [Batch*Dates*Families, Features] → [Batch*Dates*Families, 16]
        encoded = self.family_encoder(x)

        # Reshape back: [Batch*Dates*Families, 16] → [Batch, Dates, Families*16]
        encoded = encoded.reshape(batch_size, num_dates, num_families, -1)
        encoded = encoded.reshape(
            batch_size, num_dates, -1
        )  # Flatten: [Batch, Dates, 528]

        # Step 2: LSTM processing (shared across all stores)
        lstm_out, _ = self.lstm(encoded)  # [Batch, Dates, Hidden_Dim]

        # Extract last timestep (prediction for next day)
        last_hidden = lstm_out[:, -1, :]  # [Batch, Hidden_Dim]

        # Step 3: Store-specific adaptation
        store_idx = store_nbr - 1  # Convert 1-indexed to 0-indexed
        adapted = self.store_adapters[store_idx](last_hidden)  # [Batch, Hidden_Dim]

        # Step 4: Predict family sales
        predictions = self.family_predictor(adapted)  # [Batch, Families]

        return predictions

    def get_model_size(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RMSLELoss(nn.Module):
    """Root Mean Squared Logarithmic Error loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [Batch, Families] - predicted sales (log-transformed)
            actual: [Batch, Families] - actual sales (log-transformed)

        Returns:
            loss: Scalar RMSLE loss
        """
        # Data is already log-transformed, so we use MSE on log scale
        # This is equivalent to RMSLE on original scale
        return torch.sqrt(F.mse_loss(pred, actual))


def test_model():
    """Test model with dummy data (v4.1d - no feature compression)."""
    print("=" * 60)
    print("Testing Meta-Learning Multi-Task LSTM v4.1d")
    print("=" * 60)

    # Create model
    model = MetaLearningMultiTaskLSTM(
        num_features=32,  # v4.1d: All features used directly
        num_families=33,
        num_stores=54,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
    )

    print(f"\nModel size: {model.get_model_size():,} parameters")

    # Test forward pass
    batch_size = 4
    num_dates = 6  # v4.1b: window_size=7, so num_dates = 6
    num_features = 32  # v4.1d: All features
    num_families = 33

    X = torch.randn(batch_size, num_dates, num_features, num_families)
    store_nbr = 1
    print(f"\nInput shape:  {X.shape}")
    print(f"Store number: {store_nbr}")

    predictions = model(X, store_nbr)

    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Test loss
    criterion = RMSLELoss()
    y = torch.randn(batch_size, num_families)
    loss = criterion(predictions, y)

    print(f"\nLoss: {loss.item():.4f}")

    print("\n✅ Model test passed!")


# Removed test_model() function - testing now done via train_model.py
