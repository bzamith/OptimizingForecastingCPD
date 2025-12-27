"""MinMax Scaler implementation using sklearn's MinMaxScaler."""

from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from src.scaler.base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """MinMax scaler that scales features to a given range (default 0-1).

    This scaler wraps sklearn's MinMaxScaler and applies the transformation
    only to the specified variables.

    Attributes:
        variables (List[str]): List of variables to be scaled.
        scaler: Sklearn's MinMaxScaler instance.
    """

    def __init__(self, variables: List[str]):
        """Initialize the MinMaxScaler.

        Args:
            variables (List[str]): List of variables to be scaled.
        """
        super().__init__(variables)
        self.scaler = SklearnMinMaxScaler()

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
