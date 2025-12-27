"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from src.utils import _wape, get_error_results


class TestWape:
    """Tests for WAPE (Weighted Absolute Percentage Error) calculation."""

    def test_wape_perfect_prediction(self):
        """Test WAPE with perfect predictions (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        wape = _wape(y_true, y_pred)

        assert wape == 0.0

    def test_wape_basic_calculation(self):
        """Test WAPE with simple known values."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        wape = _wape(y_true, y_pred)

        # Expected: (10 + 10 + 10) / (100 + 200 + 300) = 30 / 600 = 0.05
        assert abs(wape - 0.05) < 1e-10

    def test_wape_with_negative_values(self):
        """Test WAPE with negative true values."""
        y_true = np.array([-100.0, -200.0, -300.0])
        y_pred = np.array([-110.0, -190.0, -310.0])

        wape = _wape(y_true, y_pred)

        # Expected: (10 + 10 + 10) / (100 + 200 + 300) = 30 / 600 = 0.05
        assert abs(wape - 0.05) < 1e-10

    def test_wape_zero_denominator(self):
        """Test WAPE when all true values are zero (should return NaN)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        wape = _wape(y_true, y_pred)

        assert np.isnan(wape)

    def test_wape_with_lists(self):
        """Test that WAPE works with lists (should convert to arrays)."""
        y_true = [10.0, 20.0, 30.0]
        y_pred = [11.0, 19.0, 31.0]

        wape = _wape(y_true, y_pred)

        # Expected: (1 + 1 + 1) / (10 + 20 + 30) = 3 / 60 = 0.05
        assert abs(wape - 0.05) < 1e-10

    def test_wape_single_value(self):
        """Test WAPE with single value."""
        y_true = np.array([100.0])
        y_pred = np.array([90.0])

        wape = _wape(y_true, y_pred)

        # Expected: 10 / 100 = 0.1
        assert abs(wape - 0.1) < 1e-10

    def test_wape_large_errors(self):
        """Test WAPE with predictions far from true values."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([0.0, 0.0, 0.0])

        wape = _wape(y_true, y_pred)

        # Expected: (10 + 20 + 30) / (10 + 20 + 30) = 1.0
        assert abs(wape - 1.0) < 1e-10


class TestGetErrorResults:
    """Tests for get_error_results function."""

    def test_get_error_results_perfect_prediction(self):
        """Test error results with perfect predictions."""
        y_true = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y_pred = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})

        results = get_error_results(y_true, y_pred, ["var1"])

        # With perfect predictions, all errors should be 0 (except R2 which is 1.0)
        assert results["Avg_MAE"] == 0.0
        assert results["Avg_MSE"] == 0.0
        assert results["Avg_RMSE"] == 0.0
        assert results["Avg_WAPE"] == 0.0
        assert results["var1_MAE"] == 0.0
        assert results["var1_MSE"] == 0.0
        assert results["var1_RMSE"] == 0.0
        assert results["var1_WAPE"] == 0.0

    def test_get_error_results_single_variable(self):
        """Test error results with single variable."""
        y_true = pd.DataFrame({"var1": [10.0, 20.0, 30.0, 40.0]})
        y_pred = pd.DataFrame({"var1": [11.0, 19.0, 31.0, 39.0]})

        results = get_error_results(y_true, y_pred, ["var1"])

        # Check that all expected keys exist
        assert "Avg_MAPE" in results
        assert "Avg_MAE" in results
        assert "Avg_MSE" in results
        assert "Avg_RMSE" in results
        assert "Avg_R2" in results
        assert "Avg_WAPE" in results
        assert "var1_MAPE" in results
        assert "var1_MAE" in results
        assert "var1_MSE" in results
        assert "var1_RMSE" in results
        assert "var1_R2" in results
        assert "var1_WAPE" in results

        # MAE should be 1.0 (average of [1, 1, 1, 1])
        assert results["Avg_MAE"] == 1.0
        assert results["var1_MAE"] == 1.0

        # MSE should be 1.0 (all errors are 1, so 1^2 = 1)
        assert results["Avg_MSE"] == 1.0
        assert results["var1_MSE"] == 1.0

    def test_get_error_results_multiple_variables(self):
        """Test error results with multiple variables."""
        y_true = pd.DataFrame({"var1": [10.0, 20.0, 30.0], "var2": [100.0, 200.0, 300.0]})
        y_pred = pd.DataFrame({"var1": [11.0, 19.0, 31.0], "var2": [110.0, 190.0, 310.0]})

        results = get_error_results(y_true, y_pred, ["var1", "var2"])

        # Check overall metrics exist
        assert "Avg_MAPE" in results
        assert "Avg_MAE" in results
        assert "Avg_MSE" in results
        assert "Avg_RMSE" in results
        assert "Avg_R2" in results
        assert "Avg_WAPE" in results

        # Check per-variable metrics exist for var1
        assert "var1_MAPE" in results
        assert "var1_MAE" in results
        assert "var1_MSE" in results
        assert "var1_RMSE" in results
        assert "var1_R2" in results
        assert "var1_WAPE" in results

        # Check per-variable metrics exist for var2
        assert "var2_MAPE" in results
        assert "var2_MAE" in results
        assert "var2_MSE" in results
        assert "var2_RMSE" in results
        assert "var2_R2" in results
        assert "var2_WAPE" in results

        # Total number of keys should be 6 (avg) + 6*2 (per variable) = 18
        assert len(results) == 18

    def test_get_error_results_known_values(self):
        """Test error results with known expected values."""
        y_true = pd.DataFrame({"var1": [100.0, 200.0, 300.0, 400.0]})
        y_pred = pd.DataFrame({"var1": [90.0, 210.0, 290.0, 410.0]})

        results = get_error_results(y_true, y_pred, ["var1"])

        # MAE = (10 + 10 + 10 + 10) / 4 = 10.0
        assert results["Avg_MAE"] == 10.0
        assert results["var1_MAE"] == 10.0

        # MSE = (100 + 100 + 100 + 100) / 4 = 100.0
        assert results["Avg_MSE"] == 100.0
        assert results["var1_MSE"] == 100.0

        # RMSE = sqrt(100) = 10.0
        assert results["Avg_RMSE"] == 10.0
        assert results["var1_RMSE"] == 10.0

    def test_get_error_results_with_numpy_arrays(self):
        """Test that function works when DataFrames contain numpy arrays."""
        y_true = pd.DataFrame({"var1": np.array([1.0, 2.0, 3.0])})
        y_pred = pd.DataFrame({"var1": np.array([1.5, 2.5, 3.5])})

        results = get_error_results(y_true, y_pred, ["var1"])

        # MAE should be 0.5
        assert results["Avg_MAE"] == 0.5
        assert results["var1_MAE"] == 0.5

    def test_get_error_results_r2_score(self):
        """Test R2 score calculation."""
        # Perfect linear relationship
        y_true = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y_pred = pd.DataFrame({"var1": [1.0, 2.0, 3.0, 4.0, 5.0]})

        results = get_error_results(y_true, y_pred, ["var1"])

        # R2 should be 1.0 for perfect predictions
        assert results["Avg_R2"] == 1.0
        assert results["var1_R2"] == 1.0

    def test_get_error_results_metrics_are_floats(self):
        """Test that all metrics are numeric (float)."""
        y_true = pd.DataFrame({"var1": [10.0, 20.0, 30.0]})
        y_pred = pd.DataFrame({"var1": [11.0, 19.0, 31.0]})

        results = get_error_results(y_true, y_pred, ["var1"])

        # All values should be numeric
        for key, value in results.items():
            assert isinstance(value, (int, float, np.number))

    def test_get_error_results_three_variables(self):
        """Test error results with three variables."""
        y_true = pd.DataFrame(
            {"var1": [1.0, 2.0, 3.0], "var2": [10.0, 20.0, 30.0], "var3": [100.0, 200.0, 300.0]}
        )
        y_pred = pd.DataFrame(
            {"var1": [1.1, 2.1, 3.1], "var2": [10.1, 20.1, 30.1], "var3": [100.1, 200.1, 300.1]}
        )

        results = get_error_results(y_true, y_pred, ["var1", "var2", "var3"])

        # Should have 6 overall + 6*3 per-variable = 24 metrics
        assert len(results) == 24

        # Check all three variables have their metrics
        for var in ["var1", "var2", "var3"]:
            assert f"{var}_MAPE" in results
            assert f"{var}_MAE" in results
            assert f"{var}_MSE" in results
            assert f"{var}_RMSE" in results
            assert f"{var}_R2" in results
            assert f"{var}_WAPE" in results
