"""Unit tests for change point detection functionality."""

import numpy as np
import pandas as pd
import pytest

from src.cpd import (
    BaseCPDDetector,
    BinSegCPDDetector,
    BottomUpCPDDetector,
    CPDCostFunction,
    CPDDetectorFactory,
    CPDMethod,
    FixedPercCPDDetector,
    WindowCPDDetector,
)


class TestCPDCostFunction:
    """Tests for CPDCostFunction enum."""

    def test_cost_function_from_str(self):
        """Test converting string to CPDCostFunction."""
        assert CPDCostFunction.from_str("L1") == CPDCostFunction.L1
        assert CPDCostFunction.from_str("l2") == CPDCostFunction.L2
        assert CPDCostFunction.from_str("Normal") == CPDCostFunction.NORMAL
        assert CPDCostFunction.from_str("rbf") == CPDCostFunction.RBF

    def test_cost_function_from_str_fixed_cut(self):
        """Test converting fixed cut strings."""
        assert CPDCostFunction.from_str("Fixed_Cut_0.0") == CPDCostFunction.FC0
        assert CPDCostFunction.from_str("Fixed_Cut_0.5") == CPDCostFunction.FC5
        assert CPDCostFunction.from_str("fixed_cut_0.9") == CPDCostFunction.FC9

    def test_cost_function_from_str_invalid(self):
        """Test invalid cost function string."""
        with pytest.raises(ValueError, match="Invalid cost function"):
            CPDCostFunction.from_str("invalid_cost")

    def test_cost_function_list_available(self):
        """Test listing available cost functions."""
        cost_functions = CPDCostFunction.list_available()

        assert "L1" in cost_functions
        assert "L2" in cost_functions
        assert "Normal" in cost_functions
        assert "Fixed_Cut_0.5" in cost_functions


class TestCPDMethod:
    """Tests for CPDMethod enum."""

    def test_method_from_str(self):
        """Test converting string to CPDMethod."""
        assert CPDMethod.from_str("Window") == CPDMethod.WINDOW
        assert CPDMethod.from_str("bin_seg") == CPDMethod.BIN_SEG
        assert CPDMethod.from_str("BOTTOM_UP") == CPDMethod.BOTTOM_UP
        assert CPDMethod.from_str("fixed_perc") == CPDMethod.FIXED_PERC

    def test_method_from_str_invalid(self):
        """Test invalid method string."""
        with pytest.raises(ValueError, match="Invalid change point method"):
            CPDMethod.from_str("invalid_method")

    def test_method_list_available(self):
        """Test listing available methods."""
        methods = CPDMethod.list_available()

        assert "Window" in methods
        assert "Bin_Seg" in methods
        assert "Bottom_Up" in methods
        assert "Fixed_Perc" in methods


class TestCPDDetectorFactory:
    """Tests for CPDDetectorFactory."""

    def test_create_window_detector(self):
        """Test creating Window detector."""
        detector = CPDDetectorFactory.create_detector(CPDMethod.WINDOW, CPDCostFunction.L1)

        assert isinstance(detector, WindowCPDDetector)
        assert detector.cost_function == CPDCostFunction.L1

    def test_create_binseg_detector(self):
        """Test creating BinSeg detector."""
        detector = CPDDetectorFactory.create_detector(CPDMethod.BIN_SEG, CPDCostFunction.L2)

        assert isinstance(detector, BinSegCPDDetector)
        assert detector.cost_function == CPDCostFunction.L2

    def test_create_bottomup_detector(self):
        """Test creating BottomUp detector."""
        detector = CPDDetectorFactory.create_detector(
            CPDMethod.BOTTOM_UP, CPDCostFunction.NORMAL
        )

        assert isinstance(detector, BottomUpCPDDetector)
        assert detector.cost_function == CPDCostFunction.NORMAL

    def test_create_fixedperc_detector(self):
        """Test creating FixedPerc detector."""
        detector = CPDDetectorFactory.create_detector(CPDMethod.FIXED_PERC, CPDCostFunction.FC5)

        assert isinstance(detector, FixedPercCPDDetector)
        assert detector.cost_function == CPDCostFunction.FC5

    def test_get_detector_class(self):
        """Test getting detector class."""
        detector_class = CPDDetectorFactory.get_detector_class(CPDMethod.WINDOW)

        assert detector_class == WindowCPDDetector

    def test_list_available(self):
        """Test listing available detector methods."""
        methods = CPDDetectorFactory.list_available()

        assert CPDMethod.WINDOW in methods
        assert CPDMethod.BIN_SEG in methods
        assert CPDMethod.BOTTOM_UP in methods
        assert CPDMethod.FIXED_PERC in methods

    def test_create_detector_invalid_method(self):
        """Test creating detector with invalid method raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeMethod:
            value = "FAKE_METHOD"

        with pytest.raises(ValueError, match="Unknown change point method"):
            CPDDetectorFactory.create_detector(FakeMethod, CPDCostFunction.L1)

    def test_get_detector_class_invalid_method(self):
        """Test getting detector class with invalid method raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeMethod:
            value = "FAKE_METHOD"

        with pytest.raises(ValueError, match="Unknown change point method"):
            CPDDetectorFactory.get_detector_class(FakeMethod)


class TestBaseCPDDetector:
    """Tests for BaseCPDDetector."""

    def test_get_stack_single_variable(self):
        """Test stacking single variable."""
        df = pd.DataFrame({"var1": [1, 2, 3, 4, 5]})

        # Create a concrete detector to test base class method
        detector = WindowCPDDetector(CPDCostFunction.L1)
        stacked = detector.get_stack(df, ["var1"])

        assert stacked.shape == (5, 1)
        np.testing.assert_array_equal(stacked.flatten(), np.array([1, 2, 3, 4, 5]))

    def test_get_stack_multiple_variables(self):
        """Test stacking multiple variables."""
        df = pd.DataFrame({"var1": [1, 2, 3], "var2": [4, 5, 6], "var3": [7, 8, 9]})

        detector = WindowCPDDetector(CPDCostFunction.L1)
        stacked = detector.get_stack(df, ["var1", "var2", "var3"])

        assert stacked.shape == (3, 3)
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        np.testing.assert_array_equal(stacked, expected)

    def test_apply_change_point(self):
        """Test applying change point to truncate DataFrame."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "var1": range(100)})

        detector = WindowCPDDetector(CPDCostFunction.L1)
        truncated = detector.apply_change_point(df, 30)

        assert len(truncated) == 70
        assert truncated["var1"].iloc[0] == 30
        assert truncated["var1"].iloc[-1] == 99

    def test_apply_change_point_zero(self):
        """Test applying change point at index 0."""
        df = pd.DataFrame({"var1": range(100)})

        detector = WindowCPDDetector(CPDCostFunction.L1)
        truncated = detector.apply_change_point(df, 0)

        assert len(truncated) == 100

    def test_apply_change_point_out_of_range(self):
        """Test that applying change point out of range raises error."""
        df = pd.DataFrame({"var1": range(100)})

        detector = WindowCPDDetector(CPDCostFunction.L1)

        with pytest.raises(AssertionError, match="out of dataframe range"):
            detector.apply_change_point(df, 100)


class TestFixedPercCPDDetector:
    """Tests for FixedPercCPDDetector."""

    def test_fixed_perc_initialization(self):
        """Test initialization with valid fixed cut."""
        detector = FixedPercCPDDetector(CPDCostFunction.FC5)

        assert detector.fixed_percentage == 0.5

    def test_fixed_perc_initialization_zero(self):
        """Test initialization with 0.0 cut."""
        detector = FixedPercCPDDetector(CPDCostFunction.FC0)

        assert detector.fixed_percentage == 0.0

    def test_fixed_perc_initialization_nine(self):
        """Test initialization with 0.9 cut."""
        detector = FixedPercCPDDetector(CPDCostFunction.FC9)

        assert detector.fixed_percentage == 0.9

    def test_fixed_perc_invalid_cost_function(self):
        """Test that non-fixed-cut cost function raises error."""
        with pytest.raises(AssertionError, match="Expected Fixed_Cut"):
            FixedPercCPDDetector(CPDCostFunction.L1)

    def test_find_change_point_50_percent(self):
        """Test finding change point at 50%."""
        df = pd.DataFrame({"var1": range(100)})

        detector = FixedPercCPDDetector(CPDCostFunction.FC5)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        assert change_point == 50
        assert change_point_perc == 0.5

    def test_find_change_point_30_percent(self):
        """Test finding change point at 30%."""
        df = pd.DataFrame({"var1": range(100)})

        detector = FixedPercCPDDetector(CPDCostFunction.FC3)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        assert change_point == 30
        assert change_point_perc == 0.3

    def test_find_change_point_odd_length(self):
        """Test with odd-length DataFrame."""
        df = pd.DataFrame({"var1": range(99)})

        detector = FixedPercCPDDetector(CPDCostFunction.FC5)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        assert change_point == 49  # floor(99 * 0.5)
        assert change_point_perc == 0.5


class TestWindowCPDDetector:
    """Tests for WindowCPDDetector."""

    def test_window_detector_initialization(self):
        """Test Window detector initialization."""
        detector = WindowCPDDetector(CPDCostFunction.L1)

        assert detector.cost_function == CPDCostFunction.L1
        assert detector.method is not None

    def test_window_detector_different_costs(self):
        """Test Window detector with different cost functions."""
        for cost_func in [CPDCostFunction.L1, CPDCostFunction.L2, CPDCostFunction.NORMAL]:
            detector = WindowCPDDetector(cost_func)
            assert detector.cost_function == cost_func

    def test_find_change_point_synthetic_data(self):
        """Test change point detection with synthetic data that has clear change."""
        # Create data with clear regime change at index 50
        np.random.seed(42)
        data_before = np.random.normal(0, 1, 50)
        data_after = np.random.normal(10, 1, 50)
        data = np.concatenate([data_before, data_after])

        df = pd.DataFrame({"var1": data})

        detector = WindowCPDDetector(CPDCostFunction.L2)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        # Change point should be detected (verify method returns valid results)
        assert 0 <= change_point <= len(df)
        assert isinstance(change_point_perc, float)
        assert 0 <= change_point_perc <= 100


class TestBinSegCPDDetector:
    """Tests for BinSegCPDDetector."""

    def test_binseg_detector_initialization(self):
        """Test BinSeg detector initialization."""
        detector = BinSegCPDDetector(CPDCostFunction.L2)

        assert detector.cost_function == CPDCostFunction.L2
        assert detector.method is not None

    def test_find_change_point_synthetic_data(self):
        """Test change point detection with synthetic data."""
        # Create data with clear regime change
        data_before = np.random.normal(0, 1, 50)
        data_after = np.random.normal(10, 1, 50)
        data = np.concatenate([data_before, data_after])

        df = pd.DataFrame({"var1": data})

        detector = BinSegCPDDetector(CPDCostFunction.L2)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        # Change point should be detected around index 50
        assert 40 <= change_point <= 60
        assert isinstance(change_point_perc, float)


class TestBottomUpCPDDetector:
    """Tests for BottomUpCPDDetector."""

    def test_bottomup_detector_initialization(self):
        """Test BottomUp detector initialization."""
        detector = BottomUpCPDDetector(CPDCostFunction.NORMAL)

        assert detector.cost_function == CPDCostFunction.NORMAL
        assert detector.method is not None

    def test_find_change_point_synthetic_data(self):
        """Test change point detection with synthetic data."""
        # Create data with clear regime change
        data_before = np.random.normal(0, 1, 50)
        data_after = np.random.normal(10, 1, 50)
        data = np.concatenate([data_before, data_after])

        df = pd.DataFrame({"var1": data})

        detector = BottomUpCPDDetector(CPDCostFunction.L2)
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        # Change point should be detected around index 50
        assert 40 <= change_point <= 60
        assert isinstance(change_point_perc, float)


class TestCPDIntegration:
    """Integration tests for complete CPD workflow."""

    def test_factory_creates_correct_detector_types(self):
        """Test that factory creates correct detector instances."""
        test_cases = [
            (CPDMethod.WINDOW, WindowCPDDetector),
            (CPDMethod.BIN_SEG, BinSegCPDDetector),
            (CPDMethod.BOTTOM_UP, BottomUpCPDDetector),
            (CPDMethod.FIXED_PERC, FixedPercCPDDetector),
        ]

        for method, expected_class in test_cases:
            if method == CPDMethod.FIXED_PERC:
                detector = CPDDetectorFactory.create_detector(method, CPDCostFunction.FC5)
            else:
                detector = CPDDetectorFactory.create_detector(method, CPDCostFunction.L1)

            assert isinstance(detector, expected_class)

    def test_complete_cpd_workflow(self):
        """Test complete CPD workflow: create, detect, apply."""
        # Create synthetic data
        df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=200), "var1": range(200)}
        )

        # Create detector
        detector = CPDDetectorFactory.create_detector(
            CPDMethod.FIXED_PERC, CPDCostFunction.FC5
        )

        # Find change point
        change_point, change_point_perc = detector.find_change_point(df, ["var1"])

        # Apply change point
        reduced_df = detector.apply_change_point(df, change_point)

        # Verify results
        assert change_point == 100
        assert change_point_perc == 0.5
        assert len(reduced_df) == 100
        assert reduced_df["var1"].iloc[0] == 100

    def test_multivariate_change_point_detection(self):
        """Test change point detection with multiple variables."""
        # Create multivariate data with regime change
        np.random.seed(42)
        n = 100
        data_before = np.random.multivariate_normal([0, 0, 0], np.eye(3), 50)
        data_after = np.random.multivariate_normal([10, 10, 10], np.eye(3), 50)
        data = np.vstack([data_before, data_after])

        df = pd.DataFrame(data, columns=["var1", "var2", "var3"])

        detector = WindowCPDDetector(CPDCostFunction.L2)
        change_point, change_point_perc = detector.find_change_point(
            df, ["var1", "var2", "var3"]
        )

        # Change point should be detected (verify method returns valid results)
        assert 0 <= change_point <= len(df)
        assert isinstance(change_point_perc, float)
        assert 0 <= change_point_perc <= 100
