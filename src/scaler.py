from typing import List

import pandas as pd

from sklearn import preprocessing as pp


class Scaler:
    """A scaler for scaling and descaling data using StandardScaler.

    This class wraps scikit-learn's StandardScaler to provide methods for fitting,
    transforming, and inversely transforming a DataFrame based on a list of variables.

    Attributes:
        scaler (StandardScaler): An instance of StandardScaler from sklearn.preprocessing.
        variables (List[str]): List of variables to be scaled.
    """

    scaler = pp.StandardScaler()

    def __init__(self, variables: List[str]):
        """Initialize the Scaler.

        Args:
            variables (List[str]): List of variables to be scaled.
        """
        self.variables = variables

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
