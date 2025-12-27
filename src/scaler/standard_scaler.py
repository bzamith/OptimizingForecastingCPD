"""Standard Scaler implementation using sklearn's StandardScaler."""

from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from src.scaler.base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    """Standard scaler that standardizes features by removing mean and scaling to unit variance.

    This scaler wraps sklearn's StandardScaler and applies the transformation
    only to the specified variables.

    Attributes:
        variables (List[str]): List of variables to be scaled.
        scaler: Sklearn's StandardScaler instance.
    """

    def __init__(self, variables: List[str]):
        """Initialize the StandardScaler.

        Args:
            variables (List[str]): List of variables to be scaled.
        """
        super().__init__(variables)
        self.scaler = SklearnStandardScaler()

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the scaler to the provided data.

        Args:
            data (pd.DataFrame): The input data containing columns specified in self.variables.

        Returns:
            None
        """
        self.scaler.fit(data[self.variables])

    def fit_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler to the data and transform the specified variables.

        Args:
            data (pd.DataFrame): The input data to be fitted and scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.fit_transform(data[self.variables])
        return data_output

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the specified variables in the DataFrame using the pre-fitted scaler.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.transform(data_output[self.variables])
        return data_output

    def descale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Revert the scaling transformation on the specified variables.

        Args:
            data (pd.DataFrame): The DataFrame containing the scaled data.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables descaled.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.inverse_transform(data_output[self.variables])
        return data_output
