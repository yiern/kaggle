# Titanic: Machine Learning from Disaster

[Competition Page](https://www.kaggle.com/competitions/titanic) | [Kaggle Score: 0.76076](https://www.kaggle.com/competitions/titanic/leaderboard)

Binary classification predicting passenger survival on the RMS Titanic using XGBoost with Optuna hyperparameter tuning.

---

## 🎯 Strategy Overview

| Problem | Solution | Why It Works |
|---------|----------|--------------|
| Missing data (Age, Embarked, Fare, Cabin) | Group-based imputation (Age by Pclass+Sex, mode for Embarked, median by Pclass for Fare, HasCabin binary) | Preserves relationships between variables and avoids bias from simple mean/median imputation |
| Non-linear relationships between features and survival | Gradient boosting with XGBoost | Captures complex patterns and interactions that linear models miss |
| High-dimensional hyperparameter space | Optuna with TPESampler (100 trials) | Efficiently explores parameter space using Bayesian optimization |
| Feature sparsity | Feature engineering (Title, FamilySize, IsAlone, FarePerPerson, HasCabin) | Creates meaningful predictors from raw data that improve model performance |

**Core Approach**: XGBoost classifier with automated hyperparameter tuning via Optuna, combined with thoughtful feature engineering and robust missing data handling.

---

## 🏗️ Model Architecture

```
┌─────────────────┐
│  Raw Data (CSV) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing Pipeline      │
│  • Null imputation           │
│  • Feature engineering       │
│  • Categorical encoding      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Train/Validation Split      │
│  (80/20, stratified)         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  XGBoost Classifier          │
│  • 10-fold CV                │
│  • Optuna hyperparam tuning  │
│  • 100 trials (TPESampler)   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Predictions & Evaluation    │
│  • Confusion matrix          │
│  • Feature importance        │
│  • Learning curves           │
└─────────────────────────────┘
```

### XGBoost Hyperparameters

| Parameter | Description | Tuning Range |
|-----------|-------------|--------------|
| `n_estimators` | Number of boosting rounds | 50 - 500 |
| `max_depth` | Maximum tree depth | 3 - 10 |
| `learning_rate` | Step size shrinkage | 0.01 - 0.3 |
| `subsample` | Fraction of samples for each tree | 0.5 - 1.0 |
| `colsample_bytree` | Fraction of features for each tree | 0.5 - 1.0 |
| `min_child_weight` | Minimum sum of instance weight | 1 - 10 |
| `gamma` | Minimum loss reduction | 0 - 5 |
| `reg_alpha` | L1 regularization | 0 - 1 |
| `reg_lambda` | L2 regularization | 0 - 1 |

*All hyperparameters tuned via Optuna with TPESampler*

---

## 📅 Version History & Timeline

| Version | Date | Changes | Score |
|---------|------|---------|-------|
| **v1.0** | Feb 2026 | Initial release with XGBoost + Optuna | **0.76076** |

---

## 📊 Feature Engineering Pipeline

### 1. Null Handling

```python
# Age: Median by Pclass + Sex groups
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: Mode imputation
df["Embarked"] = df["Embarked"].fillna("S")

# Fare: Median by Pclass
df["Fare"] = df.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median())
)

# Cabin: Binary feature (HasCabin)
df["HasCabin"] = df["Cabin"].notna().astype(int)
df = df.drop("Cabin", axis=1)
```

### 2. Feature Creation

```python
# Extract title from Name
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", 
               "Major", "Rev", "Sir", "Jonkheer", "Dona"]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
```

### 3. Categorical Encoding

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Title"] = encoder.fit_transform(df["Title"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])
```

**Final Feature Count**: 12 features (7 base + 5 engineered)

---

## 🎓 Training Strategy

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
```

### Hyperparameter Optimization with Optuna

```python
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
    }
    
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy").mean()
    return score

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100)
```

### Key Training Components

- **Metric**: Accuracy (primary), with confusion matrix analysis
- **Validation**: 10-fold stratified cross-validation
- **Optimization**: Optuna with TPESampler (100 trials)
- **Random State**: 42 (for reproducibility)

---

## 📁 Project Structure

```
.
├── titanic/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── models/
│       ├── __init__.py
│       └── xgboost_model.py
├── models/
│   ├── xgboost.pkl
│   └── xgboost_best.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── hyperparameter_tuning.png
│   └── learning_curve_baseline.png
├── tests/
│   └── test_feature_engineering.py
├── train.py
├── predict.py
├── AGENTS.md
└── README.md
```

---

## ⚙️ Configuration

Key configuration parameters in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RANDOM_STATE` | 42 | Random seed for reproducibility |
| `TEST_SIZE` | 0.2 | Train/validation split ratio |
| `CV_FOLDS` | 10 | Number of cross-validation folds |
| `OPTUNA_TRIALS` | 100 | Number of Optuna optimization trials |
| `OPTUNA_TIMEOUT` | 600 | Optuna optimization timeout (seconds) |

---

## 🔑 Key Learnings

### What Worked ✅

- **Optuna for hyperparameter tuning**: Automated 100-trial Bayesian optimization efficiently explored the hyperparameter space, finding configurations that outperformed manual tuning
- **XGBoost's gradient boosting**: Captured non-linear patterns and feature interactions that linear models (e.g., Logistic Regression) missed
- **Feature engineering**: Title extraction (Mr, Mrs, Miss, Rare) proved highly predictive; FamilySize and IsAlone captured social dynamics
- **Group-based imputation**: Filling missing Age values based on Pclass and Sex groups preserved demographic patterns better than global median imputation
- **Stratified cross-validation**: Ensured balanced class distribution across folds, providing reliable performance estimates

### What Didn't Work ❌

- *N/A - This is the initial version (v1.0)*

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Kaggle Submission Score** | **0.76076** |
| Training Samples | 891 |
| Test Samples | 418 |
| Features | 12 (7 base + 5 engineered) |
| Model | XGBoost with Optuna-tuned hyperparameters |
| Validation Strategy | 10-fold stratified cross-validation |

---

## 🛠️ Development Commands

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

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using XGBoost, Optuna, and scikit-learn**
