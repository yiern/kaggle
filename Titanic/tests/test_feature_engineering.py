"""Tests for preprocessing feature engineering module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_engineering import (
    handle_nulls,
    engineer_features,
    encode_features,
    preprocess,
    load_data,
)


@pytest.fixture
def sample_train():
    """Sample training data for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 1, 0],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": [
                "Braund, Mr. Owen",
                "Cumings, Mrs. John",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques",
                "Allen, Mr. William",
            ],
            "Sex": ["male", "female", "female", "female", "male"],
            "Age": [22.0, 38.0, 26.0, 35.0, None],
            "SibSp": [1, 1, 0, 1, 0],
            "Parch": [0, 0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
            "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
            "Cabin": [None, "C85", None, "C123", None],
            "Embarked": ["S", "C", "S", "S", "S"],
        }
    )


@pytest.fixture
def sample_test():
    """Sample test data for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [892, 893, 894],
            "Pclass": [3, 3, 2],
            "Name": ["Doe, Mr. John", "Smith, Mrs. Jane", "Brown, Miss. Mary"],
            "Sex": ["male", "female", "female"],
            "Age": [None, 30.0, 25.0],
            "SibSp": [0, 1, 0],
            "Parch": [0, 1, 0],
            "Ticket": ["A/5. 2151", "PP 9549", "W.E.P. 1234"],
            "Fare": [8.05, 16.70, 13.00],
            "Cabin": [None, "G6", None],
            "Embarked": ["S", "S", "Q"],
        }
    )


class TestHandleNulls:
    """Tests for handle_nulls function."""

    def test_age_imputed(self, sample_train):
        """Test that Age nulls are filled with median by Pclass+Sex."""
        result = handle_nulls(sample_train)
        assert result["Age"].isnull().sum() == 0

    def test_embarked_imputed(self, sample_train):
        """Test that Embarked nulls are filled."""
        result = handle_nulls(sample_train)
        assert result["Embarked"].isnull().sum() == 0

    def test_has_cabin_created(self, sample_train):
        """Test that HasCabin binary feature is created."""
        result = handle_nulls(sample_train)
        assert "HasCabin" in result.columns
        assert result["HasCabin"].dtype in ["int64", "int32"]

    def test_cabin_dropped(self, sample_train):
        """Test that Cabin column is dropped."""
        result = handle_nulls(sample_train)
        assert "Cabin" not in result.columns


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_title_extracted(self, sample_train):
        """Test that Title is extracted from Name."""
        result = engineer_features(sample_train)
        assert "Title" in result.columns
        assert "Mr" in result["Title"].values
        assert "Mrs" in result["Title"].values
        assert "Miss" in result["Title"].values

    def test_family_size_created(self, sample_train):
        """Test that FamilySize is created."""
        result = engineer_features(sample_train)
        assert "FamilySize" in result.columns
        assert all(
            result["FamilySize"] == sample_train["SibSp"] + sample_train["Parch"] + 1
        )

    def test_is_alone_created(self, sample_train):
        """Test that IsAlone binary feature is created."""
        result = engineer_features(sample_train)
        assert "IsAlone" in result.columns
        assert result["IsAlone"].dtype in ["int64", "int32"]

    def test_fare_per_person_created(self, sample_train):
        """Test that FarePerPerson is created."""
        result = engineer_features(sample_train)
        assert "FarePerPerson" in result.columns


class TestEncodeFeatures:
    """Tests for encode_features function."""

    def test_no_string_columns(self, sample_train):
        """Test that all columns are numeric after encoding."""
        result = encode_features(sample_train)
        for col in result.columns:
            assert result[col].dtype in ["int64", "float64", "int32", "float32"], (
                f"Column {col} is still string type"
            )

    def test_name_and_ticket_dropped(self, sample_train):
        """Test that Name and Ticket columns are dropped."""
        result = encode_features(sample_train)
        assert "Name" not in result.columns
        assert "Ticket" not in result.columns


class TestPreprocess:
    """Tests for complete preprocess pipeline."""

    def test_no_nulls_in_train(self, sample_train):
        """Test that preprocessed train has no nulls."""
        result = preprocess(sample_train)
        assert result.isnull().sum().sum() == 0

    def test_no_nulls_in_test(self, sample_test):
        """Test that preprocessed test has no nulls."""
        result = preprocess(sample_test)
        assert result.isnull().sum().sum() == 0

    def test_all_numeric(self, sample_train):
        """Test that all columns are numeric."""
        result = preprocess(sample_train)
        for col in result.columns:
            assert result[col].dtype in ["int64", "float64", "int32", "float32"], (
                f"Column {col} is not numeric: {result[col].dtype}"
            )


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dataframes(self):
        """Test that load_data returns two DataFrames."""
        train, test = load_data()
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_train_has_survived(self, sample_train):
        """Test that train data includes Survived column."""
        sample_train.to_csv("titanic/train.csv", index=False)
        train, _ = load_data()
        assert "Survived" in train.columns

    def test_test_no_survived(self, sample_train, sample_test):
        """Test that test data does not include Survived column."""
        sample_train.to_csv("titanic/train.csv", index=False)
        sample_test.to_csv("titanic/test.csv", index=False)
        _, test = load_data()
        assert "Survived" not in test.columns

    def test_correct_shapes(self):
        """Test that data has expected columns."""
        train, test = load_data()
        assert train.shape[1] == 14
        assert test.shape[1] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
