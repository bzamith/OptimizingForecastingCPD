"""Unit tests for data reader utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data_reader import fill_na, read_dataset, split_train_test, split_X_y
from src.data_reader.factory import DatasetDomain, DummyDatasets


class TestFillNA:
    """Tests for the fill_na function to ensure no data leakage."""

    def test_fill_na_forward_only(self):
        """Test that forward interpolation doesn't use future data."""
        # Create DataFrame with missing value in the middle
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )

        result = fill_na(df, ["var1"], limit_direction="forward")

        # With forward interpolation, NaN should be filled with 3.0 (linear between 2 and 4)
        assert result["var1"].iloc[2] == 3.0
        assert result["var1"].isna().sum() == 0

    def test_fill_na_backward_only(self):
        """Test backward interpolation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )

        result = fill_na(df, ["var1"], limit_direction="backward")

        # Backward interpolation should still fill the middle value
        assert result["var1"].iloc[2] == 3.0
        assert result["var1"].isna().sum() == 0

    def test_fill_na_both_directions(self):
        """Test bidirectional interpolation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )

        result = fill_na(df, ["var1"], limit_direction="both")

        assert result["var1"].iloc[1] == 2.0
        assert result["var1"].iloc[3] == 4.0
        assert result["var1"].isna().sum() == 0

    def test_fill_na_removes_leading_nans(self):
        """Test that leading NaN rows are removed."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [np.nan, np.nan, 3.0, 4.0, 5.0],
            }
        )

        result = fill_na(df, ["var1"], limit_direction="forward")

        # Should remove first 2 rows with leading NaNs
        assert len(result) == 3
        assert result["var1"].iloc[0] == 3.0

    def test_fill_na_multiple_variables(self):
        """Test filling NaN for multiple variables."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [1.0, np.nan, 3.0, 4.0, 5.0],
                "var2": [10.0, 20.0, np.nan, 40.0, 50.0],
            }
        )

        result = fill_na(df, ["var1", "var2"], limit_direction="both")

        assert result["var1"].iloc[1] == 2.0
        assert result["var2"].iloc[2] == 30.0
        assert result["var1"].isna().sum() == 0
        assert result["var2"].isna().sum() == 0

    def test_fill_na_no_data_leakage(self):
        """Critical test: ensure train data doesn't use test data."""
        # Simulate train/test split scenario
        full_data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10),
                "var1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )

        # Split data
        train = full_data.iloc[:7].copy()
        test = full_data.iloc[7:].copy()

        # Fill train with forward only (no future leakage)
        train_filled = fill_na(train, ["var1"], limit_direction="forward")

        # The NaN at index 2 should be filled with value between 2 and 4
        assert train_filled["var1"].iloc[2] == 3.0

        # Verify it didn't use any information from test set
        assert len(train_filled) == 7
        assert train_filled["var1"].max() <= 7.0

    def test_fill_na_preserves_original_dataframe(self):
        """Test that fill_na doesn't modify the original DataFrame."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "var1": [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )

        original_na_count = df["var1"].isna().sum()
        result = fill_na(df, ["var1"], limit_direction="forward")

        # Original should still have NaN
        assert df["var1"].isna().sum() == original_na_count
        # Result should have no NaN
        assert result["var1"].isna().sum() == 0


class TestSplitTrainTest:
    """Tests for the split_train_test function."""

    def test_split_train_test_basic(self):
        """Test basic train/test split."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "var1": range(100)})

        train, test = split_train_test(df)

        # Default TRAIN_PERC is 0.8
        assert len(train) == 80
        assert len(test) == 20

    def test_split_train_test_temporal_order(self):
        """Test that split maintains temporal order."""
        df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=100), "var1": range(100)}
        )

        train, test = split_train_test(df)

        # Train should come before test
        assert train["var1"].max() < test["var1"].min()

    def test_split_train_test_no_overlap(self):
        """Test that there's no overlap between train and test."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100),
                "var1": range(100),
                "id": range(100),
            }
        )

        train, test = split_train_test(df)

        train_ids = set(train["id"])
        test_ids = set(test["id"])

        assert len(train_ids & test_ids) == 0  # No overlap


class TestSplitXY:
    """Tests for the split_X_y function."""

    def test_split_X_y_shapes(self):
        """Test that X and y have correct shapes."""
        # Create a simple time series
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=100), "var1": range(100)}
        )

        X, y = split_X_y(df)

        # With OBSERVATION_WINDOW=14 and FORECAST_HORIZON=7
        # n_samples = 100 - 14 - 7 + 1 = 80
        assert X.shape[1] == 14  # OBSERVATION_WINDOW
        assert X.shape[2] == 1  # 1 variable (excluding date)
        assert y.shape[1] == 7  # FORECAST_HORIZON
        assert y.shape[2] == 1  # 1 variable

    def test_split_X_y_removes_date_column(self):
        """Test that date column is removed."""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "var1": range(100),
                "var2": range(100, 200),
            }
        )

        X, y = split_X_y(df)

        # Should have 2 features (var1, var2), not 3
        assert X.shape[2] == 2
        assert y.shape[2] == 2

    def test_split_X_y_multiple_variables(self):
        """Test with multiple variables."""
        df = pd.DataFrame(
            {
                "var1": range(100),
                "var2": range(100, 200),
                "var3": range(200, 300),
            }
        )

        X, y = split_X_y(df)

        assert X.shape[2] == 3  # 3 variables
        assert y.shape[2] == 3

    def test_split_X_y_temporal_consistency(self):
        """Test that X and y are temporally consistent."""
        df = pd.DataFrame({"var1": range(50)})

        X, y = split_X_y(df)

        # First X should be [0, 1, ..., 13] (OBSERVATION_WINDOW=14)
        # First y should be [14, 15, ..., 20] (FORECAST_HORIZON=7)
        assert X[0, -1, 0] == 13
        assert y[0, 0, 0] == 14


class TestReadDataset:
    """Tests for the read_dataset function."""

    def test_read_dataset_with_enum_domain_and_dataset(self):
        """Test reading dataset with DatasetDomain and DummyDatasets enums."""
        df, variables = read_dataset(DatasetDomain.DUMMY, DummyDatasets.DUMMY)

        # Check that DataFrame is loaded
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check that variables are correct
        assert variables == ["v1", "v2"]

        # Check that DataFrame has correct columns
        assert "ds" in df.columns
        assert "v1" in df.columns
        assert "v2" in df.columns
        assert len(df.columns) == 3  # ds, v1, v2

    def test_read_dataset_with_string_domain(self):
        """Test reading dataset with string domain (tests line 82-83)."""
        df, variables = read_dataset("dummy", DummyDatasets.DUMMY)

        assert isinstance(df, pd.DataFrame)
        assert variables == ["v1", "v2"]
        assert "ds" in df.columns
        assert "v1" in df.columns
        assert "v2" in df.columns

    def test_read_dataset_with_string_dataset(self):
        """Test reading dataset with string dataset name (tests line 86-87)."""
        df, variables = read_dataset(DatasetDomain.DUMMY, "dummy")

        assert isinstance(df, pd.DataFrame)
        assert variables == ["v1", "v2"]
        assert "ds" in df.columns
        assert "v1" in df.columns
        assert "v2" in df.columns

    def test_read_dataset_with_both_strings(self):
        """Test reading dataset with both domain and dataset as strings."""
        df, variables = read_dataset("dummy", "dummy")

        assert isinstance(df, pd.DataFrame)
        assert variables == ["v1", "v2"]
        assert "ds" in df.columns

    def test_read_dataset_filters_columns(self):
        """Test that read_dataset only returns specified columns (tests line 94)."""
        df, variables = read_dataset(DatasetDomain.DUMMY, DummyDatasets.DUMMY)

        # Should only have ds + variables, nothing else
        assert len(df.columns) == 3  # ds + v1 + v2
        assert list(df.columns) == ["ds", "v1", "v2"]

    def test_read_dataset_returns_correct_types(self):
        """Test that read_dataset returns DataFrame and list (tests line 96)."""
        result = read_dataset(DatasetDomain.DUMMY, DummyDatasets.DUMMY)

        # Check it returns a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Check types
        df, variables = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(variables, list)

    def test_read_dataset_case_insensitive_domain_string(self):
        """Test reading with case-insensitive domain string."""
        df1, vars1 = read_dataset("DUMMY", DummyDatasets.DUMMY)
        df2, vars2 = read_dataset("dummy", DummyDatasets.DUMMY)
        df3, vars3 = read_dataset("DuMmY", DummyDatasets.DUMMY)

        assert vars1 == vars2 == vars3 == ["v1", "v2"]
        assert len(df1) == len(df2) == len(df3)

    def test_read_dataset_case_insensitive_dataset_string(self):
        """Test reading with case-insensitive dataset string."""
        df1, vars1 = read_dataset(DatasetDomain.DUMMY, "DUMMY")
        df2, vars2 = read_dataset(DatasetDomain.DUMMY, "dummy")
        df3, vars3 = read_dataset(DatasetDomain.DUMMY, "DuMmY")

        assert vars1 == vars2 == vars3 == ["v1", "v2"]
        assert len(df1) == len(df2) == len(df3)


class TestDatasetDomain:
    """Tests for DatasetDomain enum."""

    def test_dataset_domain_from_str(self):
        """Test converting string to DatasetDomain."""
        assert DatasetDomain.from_str("inmet") == DatasetDomain.INMET
        assert DatasetDomain.from_str("INMET") == DatasetDomain.INMET
        assert DatasetDomain.from_str("UCI") == DatasetDomain.UCI

    def test_dataset_domain_from_str_invalid(self):
        """Test invalid dataset domain string."""
        with pytest.raises(ValueError, match="Invalid dataset domain"):
            DatasetDomain.from_str("invalid_domain")

    def test_dataset_domain_list_available(self):
        """Test listing available domains."""
        domains = DatasetDomain.list_available()

        assert "inmet" in domains
        assert "uci" in domains
        assert "tcpd" in domains
        assert "autoformer" in domains
        assert "dummy" in domains
