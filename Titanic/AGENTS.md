# AGENTS.md

Agent guidelines for operating in this repository.

---

## Project Overview

**Repository**: Titanic - Machine Learning from Disaster (Kaggle Competition)
**Description**: Binary classification predicting passenger survival on the Titanic
**Language**: Python 3.11+ | **Framework**: scikit-learn, XGBoost, Optuna
**Task**: Predict `Survived` (0/1) for test set passengers
**Metric**: Accuracy

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (XGBoost with Optuna hyperparameter tuning)
python train.py

# Generate submission file
python predict.py

# Run all tests
pytest

# Run single test
pytest tests/test_feature_engineering.py

# Run tests with coverage
pytest --cov=. --cov-report=html

# Linting and formatting
ruff check . --fix
ruff format .
```

---

## Code Style Guidelines

### Imports
```python
# Standard library → Third-party → Local
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.preprocessing.feature_engineering import preprocess
```

### Type Hints (Required)
```python
def load_data(
    train_path: str | Path = "titanic/train.csv",
    test_path: str | Path = "titanic/test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess Titanic data."""
    pass
```

### Naming Conventions
- Functions: `snake_case` → `def load_train_data()`
- Classes: `PascalCase` → `class TitanicModel`
- Constants: `UPPER_SNAKE_CASE` → `RANDOM_STATE = 42`
- Variables: `snake_case` → `train_df`, `X_train`

### Logging & Error Handling
```python
import logging

logger = logging.getLogger(__name__)

data_dir = Path("./titanic")
train_path = data_dir / "train.csv"

try:
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df):,} training samples")
except FileNotFoundError:
    logger.error(f"train.csv not found at {train_path}")
    raise
```

---

## Data Preprocessing Rules

### Pipeline Order
1. Handle missing values → 2. Engineer features → 3. Encode categoricals

### Missing Values
```python
# Age: Median by Pclass + Sex groups
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: Mode imputation
df["Embarked"] = df["Embarked"].fillna("S")

# Fare: Median by Pclass
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))

# Cabin: Binary feature
df["HasCabin"] = df["Cabin"].notna().astype(int)
df = df.drop("Cabin", axis=1)
```

### Feature Engineering
```python
# Extract title from Name
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
```

### Categorical Encoding
```python
# LabelEncoder for Sex, Title, Pclass, Embarked
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df = df.drop(["Name", "Ticket"], axis=1)
```

### Prevent Data Leakage
```python
# Scalers: FIT on training only
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
```

---

## Model Configuration

### Features to Use
- Base: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Engineered: HasCabin, Title, FamilySize, IsAlone, FarePerPerson

### Training Setup
```python
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
```

### Submission Format
```python
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## Common Pitfalls

1. Never fit scalers on test data (data leakage)
2. Don't split test.csv - use for final submission only
3. Set random_state for reproducibility
4. Drop PassengerId/Name before training (except Title extraction)
5. Use stratify in train_test_split for imbalanced classes
