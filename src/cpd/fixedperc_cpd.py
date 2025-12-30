"""Fixed Percentage change point detector implementation."""

import math
from typing import List, Tuple, Union

import pandas as pd

from src.cpd.base_cpd import BaseCPDDetector
from src.cpd.factory import CPDCostFunction


class FixedPercCPDDetector(BaseCPDDetector):
    """Fixed Percentage change point detector.

    This detector doesn't actually detect change points, but instead
    uses a fixed percentage of the data as the change point location.
    Useful as a baseline comparison.

    Attributes:
        cost_function (CPDCostFunction): Should be one of the Fixed_Cut values.
        fixed_percentage (float): The fixed percentage to use (0.0 to 1.0).
    """

    def __init__(self, cost_function: CPDCostFunction):
        """Initialize the Fixed Percentage change point detector.

        Args:
            cost_function (CPDCostFunction): Must be a Fixed_Cut value.

        Raises:
            AssertionError: If cost function is not a Fixed_Cut type.
        """
        super().__init__(cost_function)

        assert cost_function.value.startswith(
            "Fixed_Cut"
        ), f"Expected Fixed_Cut for cost_function in Fixed Percentage, instead got {cost_function.value}"

        percentage = float(cost_function.value[-3:])

        assert (
            0 <= percentage <= 1
        ), f"Fixed cut value {percentage} is out of range. Must be between 0 and 1."

        self.fixed_percentage = percentage

    def find_change_point(
        self, df: Union[pd.DataFrame, pd.Series], variables: List[str]
    ) -> Tuple[int, float]:
        """Find change point using fixed percentage.

        Args:
            df (pd.DataFrame): Time series data.
            variables (List[str]): Variables to analyze (not used in this method).

        Returns:
            Tuple[int, float]: Change point index and percentage.
        """
        change_point = math.floor(df.shape[0] * self.fixed_percentage)
        return change_point, self.fixed_percentage
