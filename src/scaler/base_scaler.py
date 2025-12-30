"""Base scaler classes for data normalization.

This module contains the base class that all scaler implementations inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd


class BaseScaler(ABC):
    """Abstract base class for all scalers.

    This class provides a common interface for scaling and descaling data.

    Attributes:
        variables (List[str]): List of variables to be scaled.
    """

    def __init__(self, variables: List[str]):
        """Initialize the BaseScaler.

        Args:
            variables (List[str]): List of variables to be scaled.
        """
        self.variables = variables

    @abstractmethod
    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Fit the scaler to the provided data.

        This method must be implemented by subclasses.

        Args:
            data (pd.DataFrame): The input data containing columns specified in self.variables.

        Returns:
            None
        """
        pass

    @abstractmethod
    def fit_scale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Fit the scaler to the data and transform the specified variables.

        This method must be implemented by subclasses.

        Args:
            data (pd.DataFrame): The input data to be fitted and scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        pass

    @abstractmethod
    def scale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Scale the specified variables in the DataFrame using the pre-fitted scaler.

        This method must be implemented by subclasses.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        pass

    @abstractmethod
    def descale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Revert the scaling transformation on the specified variables.

        This method must be implemented by subclasses.

        Args:
            data (pd.DataFrame): The DataFrame containing the scaled data.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables descaled.
        """
        pass
