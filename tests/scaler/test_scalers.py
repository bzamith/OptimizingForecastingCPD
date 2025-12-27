"""Unit tests for individual scaler implementations."""

import numpy as np
import pandas as pd
import pytest

from src.scaler import MaxAbsScaler, MinMaxScaler, RobustScaler


class TestMinMaxScaler:
    """Tests for MinMaxScaler implementation."""

    def test_fit_scale_range_0_to_1(self):
        """Test that MinMaxScaler scales values to [0, 1] range."""
        df = pd.DataFrame({"var1": [0.0, 50.0, 100.0]})

        scaler = MinMaxScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Min should be 0, max should be 1
        assert scaled_df["var1"].min() == 0.0
        assert scaled_df["var1"].max() == 1.0
        assert scaled_df["var1"].iloc[1] == 0.5

    def test_fit_and_scale_separately(self):
        """Test fit and scale called separately."""
        train_df = pd.DataFrame({"var1": [0.0, 50.0, 100.0]})
        test_df = pd.DataFrame({"var1": [25.0, 75.0]})

        scaler = MinMaxScaler(variables=["var1"])
        scaler.fit(train_df)
        scaled_test = scaler.scale(test_df)

        # Values should be scaled based on train min/max
        assert scaled_test["var1"].iloc[0] == 0.25
        assert scaled_test["var1"].iloc[1] == 0.75

    def test_descale_inverse_transform(self):
        """Test that descale correctly inverts scaling."""
        df = pd.DataFrame({"var1": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = MinMaxScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)
        descaled_df = scaler.descale(scaled_df)

        # Should recover original values
        pd.testing.assert_frame_equal(descaled_df, df, check_exact=False, atol=1e-10)

    def test_minmax_preserves_non_scaled_columns(self):
        """Test that non-specified columns are preserved."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "var1": [0.0, 25.0, 50.0, 75.0, 100.0]})

        scaler = MinMaxScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Date column should be unchanged
        pd.testing.assert_series_equal(scaled_df["date"], df["date"])


class TestMaxAbsScaler:
    """Tests for MaxAbsScaler implementation."""

    def test_fit_scale_range_minus1_to_1(self):
        """Test that MaxAbsScaler scales values to [-1, 1] range."""
        df = pd.DataFrame({"var1": [-100.0, 0.0, 50.0, 100.0]})

        scaler = MaxAbsScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Max absolute value should be 1
        assert scaled_df["var1"].abs().max() == 1.0
        assert scaled_df["var1"].iloc[0] == -1.0  # -100 / 100 = -1
        assert scaled_df["var1"].iloc[1] == 0.0   # 0 / 100 = 0
        assert scaled_df["var1"].iloc[2] == 0.5   # 50 / 100 = 0.5
        assert scaled_df["var1"].iloc[3] == 1.0   # 100 / 100 = 1

    def test_fit_and_scale_separately(self):
        """Test fit and scale called separately."""
        train_df = pd.DataFrame({"var1": [-100.0, 0.0, 100.0]})
        test_df = pd.DataFrame({"var1": [-50.0, 50.0]})

        scaler = MaxAbsScaler(variables=["var1"])
        scaler.fit(train_df)
        scaled_test = scaler.scale(test_df)

        # Values should be scaled based on train max absolute value (100)
        assert scaled_test["var1"].iloc[0] == -0.5  # -50 / 100
        assert scaled_test["var1"].iloc[1] == 0.5   # 50 / 100

    def test_descale_inverse_transform(self):
        """Test that descale correctly inverts scaling."""
        df = pd.DataFrame({"var1": [-50.0, -25.0, 0.0, 25.0, 50.0]})

        scaler = MaxAbsScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)
        descaled_df = scaler.descale(scaled_df)

        # Should recover original values
        pd.testing.assert_frame_equal(descaled_df, df, check_exact=False, atol=1e-10)

    def test_maxabs_preserves_non_scaled_columns(self):
        """Test that non-specified columns are preserved."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "var1": [-100.0, 0.0, 100.0]})

        scaler = MaxAbsScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Date column should be unchanged
        pd.testing.assert_series_equal(scaled_df["date"], df["date"])


class TestRobustScaler:
    """Tests for RobustScaler implementation."""

    def test_fit_scale_robust_to_outliers(self):
        """Test that RobustScaler is robust to outliers."""
        # Data with outlier
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})

        scaler = RobustScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # The median (between 3 and 4) should map to around 0
        # The IQR-based scaling should not be heavily influenced by the outlier (100)
        median_value = scaled_df["var1"].iloc[2:4].mean()
        assert -0.5 < median_value < 0.5

    def test_fit_and_scale_separately(self):
        """Test fit and scale called separately."""
        train_df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        test_df = pd.DataFrame({"var1": [2.5, 3.5]})

        scaler = RobustScaler(variables=["var1"])
        scaler.fit(train_df)
        scaled_test = scaler.scale(test_df)

        # Just verify it runs and produces output
        assert len(scaled_test) == 2
        assert "var1" in scaled_test.columns

    def test_descale_inverse_transform(self):
        """Test that descale correctly inverts scaling."""
        df = pd.DataFrame({"var1": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = RobustScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)
        descaled_df = scaler.descale(scaled_df)

        # Should recover original values
        pd.testing.assert_frame_equal(descaled_df, df, check_exact=False, atol=1e-10)

    def test_robust_preserves_non_scaled_columns(self):
        """Test that non-specified columns are preserved."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "var1": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = RobustScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Date column should be unchanged
        pd.testing.assert_series_equal(scaled_df["date"], df["date"])

    def test_robust_with_multiple_variables(self):
        """Test RobustScaler with multiple variables."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0], "var2": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = RobustScaler(variables=["var1", "var2"])
        scaled_df = scaler.fit_scale(df)

        # Both variables should be scaled
        assert "var1" in scaled_df.columns
        assert "var2" in scaled_df.columns


class TestScalerComparison:
    """Comparative tests between different scalers."""

    def test_all_scalers_preserve_shape(self):
        """Test that all scaler types preserve DataFrame shape."""
        df = pd.DataFrame({"var1": np.random.randn(100), "var2": np.random.randn(100)})

        for scaler_class in [MinMaxScaler, MaxAbsScaler, RobustScaler]:
            scaler = scaler_class(variables=["var1", "var2"])
            scaled_df = scaler.fit_scale(df)

            assert scaled_df.shape == df.shape
            assert list(scaled_df.columns) == list(df.columns)

    def test_all_scalers_invertible(self):
        """Test that all scalers can invert their transformations."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})

        for scaler_class in [MinMaxScaler, MaxAbsScaler, RobustScaler]:
            scaler = scaler_class(variables=["var1"])
            scaled_df = scaler.fit_scale(df)
            descaled_df = scaler.descale(scaled_df)

            np.testing.assert_allclose(descaled_df["var1"].values, df["var1"].values, atol=1e-10)
