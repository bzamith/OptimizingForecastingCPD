"""Base change point detector classes.

This module contains the base class that all change point detector implementations inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.cpd.factory import CPDCostFunction


class BaseCPDDetector(ABC):
    """Abstract base class for change point detectors.

    This class provides common functionality for all change point detector implementations.

    Attributes:
        cost_function: The cost function used for detection.
    """

    def __init__(self, cost_function: "CPDCostFunction"):
        """Initialize the base change point detector.

        Args:
            cost_function (CPDCostFunction): Cost function to use.
        """
        self.cost_function = cost_function

    def get_stack(self, df: Union[pd.Series, pd.DataFrame], variables: List[str]) -> np.ndarray:
        """Stack multiple variable columns into a numpy array.

        Args:
            df (pd.DataFrame): DataFrame containing the time series data.
            variables (List[str]): List of column names to stack.

        Returns:
            np.ndarray: Vertically stacked array of shape (n_samples, n_variables).
        """
        stack_list = []
        for col in variables:
            stack_list.append(df[col].values)
        return np.vstack(stack_list).T

    @abstractmethod
    def find_change_point(
        self, df: Union[pd.Series, pd.DataFrame], variables: List[str]
    ) -> Tuple[int, float]:
        """Find the change point in the time series.

        This method must be implemented by subclasses.

        Args:
            df (pd.DataFrame): Time series data.
            variables (List[str]): Variables to analyze.

        Returns:
            Tuple[int, float]: Change point index and percentage.
        """
        pass

    def apply_change_point(
        self, df: Union[pd.Series, pd.DataFrame], change_point: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply the change point by truncating the DataFrame.

        Args:
            df (pd.DataFrame): Original DataFrame.
            change_point (int): Index where the change point was detected.

        Returns:
            pd.DataFrame: Truncated DataFrame starting from the change point.

        Raises:
            AssertionError: If change point is out of range.
        """
        assert (
            change_point < df.shape[0]
        ), f"Cut point {change_point} out of dataframe range ({len(df)})"
        return df.iloc[change_point:]
