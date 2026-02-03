# Kaggle Housing Competition - AI Agent Instructions

## Project Overview
This is a **Kaggle housing price prediction project** using the Ames Housing dataset. The goal is to predict `SalePrice` for houses based on 81 features using scikit-learn regression models.

## Architecture & Data Flow

### Core Pipeline
1. **Data Loading** → `train.csv` from `home-data-for-ml-course/`
2. **Feature Engineering** → Create derived features, drop originals to avoid multicollinearity
3. **Preprocessing** → Handle missing values, encode categoricals, remove outliers
4. **Model Training** → RandomForest & GradientBoosting with GridSearchCV
5. **Evaluation** → Compare MAE/RMSE across approaches

### Key Files
- `housing_competition.ipynb` - Main notebook with complete ML pipeline
- `home-data-for-ml-course/train.csv` - Training data (1460 houses, 81 columns)
- `ml-project-roadmap.md` - Learning roadmap and ML concepts reference

## Critical Patterns & Conventions

### Feature Engineering (Avoid Multicollinearity!)
**Always drop original columns after combining them:**
```python
# ✅ Correct pattern - prevents redundant information
X['HouseAge'] = housing_data['YrSold'] - housing_data['YearBuilt']
X['TotalSF'] = housing_data['TotalBsmtSF'] + housing_data['1stFlrSF'] + housing_data['2ndFlrSF']
X = X.drop(columns=['YearBuilt', 'YrSold', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'])
```
**Why:** Combined features capture all information; keeping originals dilutes feature importance and confuses models.

### Categorical Encoding Strategy
**Filter by cardinality before encoding:**
```python
# Only keep categoricals with 2-20 unique values
good_categoricals = [col for col, n in categorical_cardinality.items() if 2 <= n <= 20]
X_encoded = pd.get_dummies(X, columns=good_categoricals, drop_first=True)
```
**Why:** High-cardinality features (>20 categories) create noise and worsen performance. The project found that including ALL categoricals degrades MAE compared to numeric-only models.

### Missing Value Handling
- **Numeric columns:** Fill with median (not mean, to handle outliers)
- **Categorical columns:** Fill with explicit `'Missing'` category (not mode)
- **Timing:** Handle before encoding but after train/test split

### Outlier Management
The project uses IQR method (1.5 × IQR fences) but **keeps outliers** in final models:
```python
def remove_outliers(data: pd.DataFrame, column: str):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[(data[column] >= Q1-1.5*IQR) & (data[column] <= Q3+1.5*IQR)]
```
**Decision:** Outliers are legitimate luxury houses; removing them harms real-world predictions.

### Model Configuration
**RandomForest hyperparameters that work:**
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    max_features='sqrt',  # Critical for preventing overfitting with many features
    random_state=42
)
```

**GradientBoosting hyperparameters:**
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### Random State = 42 Everywhere
**Always set `random_state=42`** for:
- `train_test_split()`
- Model instantiation
- `GridSearchCV()`

**Why:** Ensures reproducible results across runs for comparing approaches.

## Evaluation Metrics

### Primary: MAE (Mean Absolute Error)
- **Interpretation:** Average dollar amount model is off by
- **Used in:** GridSearchCV scoring (`'neg_mean_absolute_error'`)
- **Typical range:** $15,000-$25,000 depending on approach

### Secondary: RMSE (Root Mean Squared Error)
- **Calculation:** `np.sqrt(mean_squared_error(y_true, y_pred))`
- **Property:** Always ≥ MAE; difference indicates outlier sensitivity
- **Interpretation:** If RMSE >> MAE, model has large prediction errors on some houses

## Common Workflows

### Running GridSearchCV for Hyperparameter Tuning
```python
param_grid = {'n_estimators': [100,200,300], 'max_depth': [10,20,30]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                    cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid.fit(X_train, y_train)
best_mae = -grid.best_score_  # Flip sign to get real MAE
```

### Comparing Numeric-Only vs With-Categoricals
The project systematically compares approaches - always preserve previous MAE in variables for comparison:
```python
mae_numeric = 18500  # Store baseline
mae_with_cats = 20100  # New approach
print(f"Impact: ${mae_with_cats - mae_numeric:+,.2f}")  # Shows degradation
```

### Visualizing Categorical Impact
Uses boxplots ordered by median price to understand feature importance:
```python
order = data.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).index
sns.boxplot(data=data, x='Neighborhood', y='SalePrice', order=order)
```

## Data Conventions

### Column Naming
- Engineered features use descriptive names: `HouseAge`, `TotalSF`, `TotalBathrooms`, `TotalPorchSF`
- Original columns: Keep pandas default naming from CSV

### Train/Test Split
- **Split ratio:** 80/20 (standard for this dataset size)
- **Always use:** `random_state=42`
- **Split BEFORE encoding** to prevent data leakage

### Handling Encoded Features
After `pd.get_dummies()`, train and validation sets may have different columns. Always align:
```python
X_train_encoded, X_valid_encoded = X_train_encoded.align(
    X_valid_encoded, join='left', axis=1, fill_value=0
)
```

## Known Issues & Decisions

1. **Categorical features degrade performance** - Including all categoricals increases MAE from ~$18K to ~$20K. Solution: Filter by cardinality.
2. **Outliers are kept** - Removing them improves training metrics but hurts real predictions.
3. **Feature importance dilution** - Original columns + engineered features split importance scores. Always drop originals.
4. **No feature scaling needed** - Tree-based models (RF, GradientBoosting) don't require normalization.

## Testing & Debugging

### Verify Feature Engineering
```python
# Check multicollinearity
assert 'YearBuilt' not in X.columns, "Should drop YearBuilt after creating HouseAge"
assert 'HouseAge' in X.columns, "HouseAge feature missing"
```

### Validate Encoding
```python
# Ensure no object columns remain
assert X_encoded.select_dtypes(include=['object']).shape[1] == 0, "Text columns remain!"
```

### Check Model Performance
Expected MAE ranges:
- Numeric-only: ~$18,000-$20,000
- With filtered categoricals: ~$17,000-$19,000
- With all categoricals (BAD): ~$20,000-$25,000

## Python Environment
- Virtual env at `.venv/` (activate: `source .venv/bin/activate`)
- Key dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
- Python version: Compatible with pandas 2.x API

## References
- Dataset: Kaggle Ames Housing dataset (`home-data-for-ml-course/`)
- Learning resource: `ml-project-roadmap.md` explains ML concepts progressively
