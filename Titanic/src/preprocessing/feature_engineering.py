"""Feature engineering utilities for Titanic dataset.

This module provides functions to preprocess the Titanic dataset for machine learning,
including null value handling, feature engineering, and categorical encoding.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the Titanic dataset.

    Args:
        df: Raw DataFrame with potential null values.

    Returns:
        DataFrame with null values handled.

    Strategies:
        - Age: Median by Pclass + Sex groups
        - Embarked: Mode imputation ('S')
        - Fare: Median by Pclass
        - Cabin: Create HasCabin binary feature, drop original
    """
    df = df.copy()

    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    df["Embarked"] = df["Embarked"].fillna("S")

    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))

    df["HasCabin"] = df["Cabin"].notna().astype(int)

    df = df.drop("Cabin", axis=1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns.

    Args:
        df: DataFrame after null handling.

    Returns:
        DataFrame with new engineered features.

    Features created:
        - Title: Extracted from Name (Mr, Mrs, Miss, etc.)
        - FamilySize: Total family members (SibSp + Parch + 1)
        - IsAlone: Binary (1 if solo traveler)
        - FarePerPerson: Adjusted fare per family member
    """
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        DataFrame with all categorical columns encoded to integers.

    Columns encoded:
        - Sex: male=0, female=1
        - Title: Mr=0, Mrs=1, Miss=2, Master=3, Rare=4
        - Pclass: 1st=0, 2nd=1, 3rd=2
        - Embarked: S=0, C=1, Q=2

    Columns dropped:
        - Name: Only used for Title extraction
        - Ticket: Not useful for modeling
    """
    df = df.copy()

    label_encoders: dict[str, LabelEncoder] = {}

    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])
    label_encoders["Sex"] = sex_encoder

    title_encoder = LabelEncoder()
    df["Title"] = title_encoder.fit_transform(df["Title"])
    label_encoders["Title"] = title_encoder

    pclass_encoder = LabelEncoder()
    df["Pclass"] = pclass_encoder.fit_transform(df["Pclass"])
    label_encoders["Pclass"] = pclass_encoder

    embarked_encoder = LabelEncoder()
    df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])
    label_encoders["Embarked"] = embarked_encoder

    df = df.drop(["Name", "Ticket"], axis=1)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Complete preprocessing pipeline for Titanic dataset.

    Args:
        df: Raw DataFrame from Titanic CSV.

    Returns:
        Fully preprocessed DataFrame ready for machine learning.

    Pipeline:
        1. Handle null values
        2. Engineer new features
        3. Encode categorical features

    Output columns:
        PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare,
        HasCabin, Title, FamilySize, IsAlone, FarePerPerson, Embarked
    """
    df = handle_nulls(df)
    df = engineer_features(df)
    df = encode_features(df)
    return df


def load_data(
    train_path: str | Path = "titanic/train.csv",
    test_path: str | Path = "titanic/test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess Titanic training and test data.

    Args:
        train_path: Path to training CSV file.
        test_path: Path to test CSV file.

    Returns:
        Tuple of (train_df, test_df) both preprocessed.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    return train_df, test_df
