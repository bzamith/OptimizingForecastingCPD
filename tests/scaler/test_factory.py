"""Unit tests for scaler functionality."""

import numpy as np
import pandas as pd
import pytest

from src.scaler import (
    MaxAbsScaler,
    MinMaxScaler,
    NoneScaler,
    RobustScaler,
    ScalerFactory,
    ScalerType,
    StandardScaler,
)


class TestScalerType:
    """Tests for ScalerType enum."""

    def test_scaler_type_from_str(self):
        """Test converting string to ScalerType."""
        assert ScalerType.from_str("standard") == ScalerType.STANDARD
        assert ScalerType.from_str("MINMAX") == ScalerType.MINMAX
        assert ScalerType.from_str("Robust") == ScalerType.ROBUST
        assert ScalerType.from_str("maxabs") == ScalerType.MAXABS
        assert ScalerType.from_str("none") == ScalerType.NONE

    def test_scaler_type_from_str_case_insensitive(self):
        """Test that scaler type is case-insensitive."""
        assert ScalerType.from_str("STANDARD") == ScalerType.STANDARD
        assert ScalerType.from_str("Standard") == ScalerType.STANDARD
        assert ScalerType.from_str("standard") == ScalerType.STANDARD

    def test_scaler_type_from_str_invalid(self):
        """Test invalid scaler type string."""
        with pytest.raises(ValueError, match="Invalid scaler type"):
            ScalerType.from_str("invalid_scaler")

    def test_scaler_type_list_available(self):
        """Test listing available scaler types."""
        scaler_types = ScalerType.list_available()

        assert "standard" in scaler_types
        assert "minmax" in scaler_types
        assert "robust" in scaler_types
        assert "maxabs" in scaler_types
        assert "none" in scaler_types
        assert len(scaler_types) == 5


class TestScalerFactory:
    """Tests for ScalerFactory."""

    def test_create_standard_scaler(self):
        """Test creating Standard scaler."""
        scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables=["var1", "var2"])

        assert isinstance(scaler, StandardScaler)
        assert scaler.variables == ["var1", "var2"]

    def test_create_minmax_scaler(self):
        """Test creating MinMax scaler."""
        scaler = ScalerFactory.create_scaler(ScalerType.MINMAX, variables=["var1"])

        assert isinstance(scaler, MinMaxScaler)
        assert scaler.variables == ["var1"]

    def test_create_robust_scaler(self):
        """Test creating Robust scaler."""
        scaler = ScalerFactory.create_scaler(ScalerType.ROBUST, variables=["var1", "var2", "var3"])

        assert isinstance(scaler, RobustScaler)
        assert scaler.variables == ["var1", "var2", "var3"]

    def test_create_maxabs_scaler(self):
        """Test creating MaxAbs scaler."""
        scaler = ScalerFactory.create_scaler(ScalerType.MAXABS, variables=["var1"])

        assert isinstance(scaler, MaxAbsScaler)
        assert scaler.variables == ["var1"]

    def test_create_none_scaler(self):
        """Test creating None scaler."""
        scaler = ScalerFactory.create_scaler(ScalerType.NONE, variables=["var1", "var2"])

        assert isinstance(scaler, NoneScaler)
        assert scaler.variables == ["var1", "var2"]

    def test_get_scaler_class(self):
        """Test getting scaler class."""
        scaler_class = ScalerFactory.get_scaler_class(ScalerType.STANDARD)

        assert scaler_class == StandardScaler

    def test_get_scaler_class_all_types(self):
        """Test getting all scaler classes."""
        test_cases = [
            (ScalerType.STANDARD, StandardScaler),
            (ScalerType.MINMAX, MinMaxScaler),
            (ScalerType.ROBUST, RobustScaler),
            (ScalerType.MAXABS, MaxAbsScaler),
            (ScalerType.NONE, NoneScaler),
        ]

        for scaler_type, expected_class in test_cases:
            scaler_class = ScalerFactory.get_scaler_class(scaler_type)
            assert scaler_class == expected_class

    def test_list_available_types(self):
        """Test listing available scaler types."""
        types = ScalerFactory.list_available_types()

        assert ScalerType.STANDARD in types
        assert ScalerType.MINMAX in types
        assert ScalerType.ROBUST in types
        assert ScalerType.MAXABS in types
        assert ScalerType.NONE in types
        assert len(types) == 5

    def test_create_scaler_invalid_type(self):
        """Test creating scaler with invalid type raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeType:
            value = "FAKE_SCALER"

        with pytest.raises(ValueError, match="Unknown scaler type"):
            ScalerFactory.create_scaler(FakeType, variables=["var1"])

    def test_get_scaler_class_invalid_type(self):
        """Test getting scaler class with invalid type raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeType:
            value = "FAKE_SCALER"

        with pytest.raises(ValueError, match="Unknown scaler type"):
            ScalerFactory.get_scaler_class(FakeType)


class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_fit_scale_single_variable(self):
        """Test fit_scale with single variable."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})

        scaler = StandardScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Mean should be close to 0
        assert abs(scaled_df["var1"].mean()) < 1e-10
        # Std should be reasonable (sklearn uses ddof=0, pandas uses ddof=1 by default)
        assert 0.9 < scaled_df["var1"].std() < 1.2

    def test_fit_scale_multiple_variables(self):
        """Test fit_scale with multiple variables."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0], "var2": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = StandardScaler(variables=["var1", "var2"])
        scaled_df = scaler.fit_scale(df)

        assert abs(scaled_df["var1"].mean()) < 1e-10
        assert abs(scaled_df["var2"].mean()) < 1e-10

    def test_scale_after_fit(self):
        """Test scale method after fitting."""
        train_df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        test_df = pd.DataFrame({"var1": [3.0, 4.0, 5.0]})

        scaler = StandardScaler(variables=["var1"])
        scaler.fit(train_df)
        scaled_test = scaler.scale(test_df)

        # Test data should be scaled using train statistics
        assert len(scaled_test) == 3

    def test_descale_inverse_transform(self):
        """Test that descale correctly inverts scaling."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0], "var2": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = StandardScaler(variables=["var1", "var2"])
        scaled_df = scaler.fit_scale(df)
        descaled_df = scaler.descale(scaled_df)

        # Should recover original values
        pd.testing.assert_frame_equal(descaled_df, df, check_exact=False, atol=1e-10)

    def test_fit_scale_preserves_original(self):
        """Test that fit_scale doesn't modify original DataFrame."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        original_values = df["var1"].copy()

        scaler = StandardScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Original should be unchanged
        pd.testing.assert_series_equal(df["var1"], original_values)
        # Scaled should be different
        assert not df["var1"].equals(scaled_df["var1"])

    def test_scale_preserves_non_scaled_columns(self):
        """Test that scaling preserves columns not in variables list."""
        df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=5), "var1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        scaler = StandardScaler(variables=["var1"])
        scaled_df = scaler.fit_scale(df)

        # Date column should be unchanged
        pd.testing.assert_series_equal(scaled_df["date"], df["date"])


class TestNoneScaler:
    """Tests for NoneScaler (identity transformation)."""

    def test_none_scaler_fit_scale(self):
        """Test that NoneScaler returns data unchanged."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0], "var2": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaler = NoneScaler(variables=["var1", "var2"])
        scaled_df = scaler.fit_scale(df)

        # Should be identical
        pd.testing.assert_frame_equal(scaled_df, df)

    def test_none_scaler_descale(self):
        """Test that NoneScaler descale returns data unchanged."""
        df = pd.DataFrame({"var1": [1.0, 2.0, 3.0]})

        scaler = NoneScaler(variables=["var1"])
        scaler.fit(df)
        scaled_df = scaler.scale(df)
        descaled_df = scaler.descale(scaled_df)

        pd.testing.assert_frame_equal(descaled_df, df)


class TestScalerIntegration:
    """Integration tests for scaler workflow."""

    def test_complete_scaling_workflow(self):
        """Test complete workflow: fit on train, scale train and test, descale."""
        # Create train and test data
        train_df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=100), "var1": range(100)}
        )
        test_df = pd.DataFrame(
            {"date": pd.date_range("2020-04-10", periods=30), "var1": range(100, 130)}
        )

        # Create scaler
        scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables=["var1"])

        # Fit on train and scale
        scaled_train = scaler.fit_scale(train_df)
        scaled_test = scaler.scale(test_df)

        # Verify scaling
        assert abs(scaled_train["var1"].mean()) < 1e-10
        assert len(scaled_test) == 30

        # Descale and verify
        descaled_train = scaler.descale(scaled_train)
        descaled_test = scaler.descale(scaled_test)

        # Compare values (ignore dtype differences as scaling converts to float)
        np.testing.assert_allclose(
            descaled_train["var1"].values, train_df["var1"].values, atol=1e-10
        )
        np.testing.assert_allclose(
            descaled_test["var1"].values, test_df["var1"].values, atol=1e-10
        )

    def test_all_scalers_preserve_shape(self):
        """Test that all scaler types preserve DataFrame shape."""
        df = pd.DataFrame(
            {"var1": np.random.randn(100), "var2": np.random.randn(100), "var3": np.random.randn(100)}
        )

        for scaler_type in ScalerType:
            scaler = ScalerFactory.create_scaler(scaler_type, variables=["var1", "var2", "var3"])
            scaled_df = scaler.fit_scale(df)

            assert scaled_df.shape == df.shape
            assert list(scaled_df.columns) == list(df.columns)

    def test_scaler_data_leakage_prevention(self):
        """Test that scaling train and test separately prevents data leakage."""
        # Create data with different distributions
        train_df = pd.DataFrame({"var1": np.random.normal(0, 1, 100)})
        test_df = pd.DataFrame({"var1": np.random.normal(10, 2, 30)})

        # Correct approach: fit on train only
        scaler = StandardScaler(variables=["var1"])
        scaler.fit(train_df)
        scaled_train = scaler.scale(train_df)
        scaled_test = scaler.scale(test_df)

        # Train should have mean ~0, std ~1
        assert abs(scaled_train["var1"].mean()) < 0.2
        # Test will NOT have mean 0 (because it uses train statistics)
        # This is correct behavior - prevents data leakage

    def test_factory_creates_correct_scaler_types(self):
        """Test that factory creates correct scaler instances."""
        test_cases = [
            (ScalerType.STANDARD, StandardScaler),
            (ScalerType.MINMAX, MinMaxScaler),
            (ScalerType.ROBUST, RobustScaler),
            (ScalerType.MAXABS, MaxAbsScaler),
            (ScalerType.NONE, NoneScaler),
        ]

        for scaler_type, expected_class in test_cases:
            scaler = ScalerFactory.create_scaler(scaler_type, variables=["var1"])
            assert isinstance(scaler, expected_class)
            assert scaler.variables == ["var1"]
