"""MaxAbs Scaler implementation using sklearn's MaxAbsScaler."""

from typing import List, Union

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler

from src.scaler.base_scaler import BaseScaler


class MaxAbsScaler(BaseScaler):
    """MaxAbs scaler that scales features by their maximum absolute value.

    This scaler scales each feature to the range [-1, 1] by dividing through the
    maximum absolute value of each feature. It wraps sklearn's MaxAbsScaler and
    applies the transformation only to the specified variables.

    Attributes:
        variables (List[str]): List of variables to be scaled.
        scaler: Sklearn's MaxAbsScaler instance.
    """

    def __init__(self, variables: List[str]):
        """Initialize the MaxAbsScaler.

        Args:
            variables (List[str]): List of variables to be scaled.
        """
        super().__init__(variables)
        self.scaler = SklearnMaxAbsScaler()

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Fit the scaler to the provided data.

        Args:
            data (pd.DataFrame): The input data containing columns specified in self.variables.

        Returns:
            None
        """
        self.scaler.fit(data[self.variables])

    def fit_scale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Fit the scaler to the data and transform the specified variables.

        Args:
            data (pd.DataFrame): The input data to be fitted and scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        # Optimize: Use assign to avoid full DataFrame copy
        scaled_values = self.scaler.fit_transform(data[self.variables])
        return data.assign(**{var: scaled_values[:, i] for i, var in enumerate(self.variables)})

    def scale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Scale the specified variables in the DataFrame using the pre-fitted scaler.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        # Optimize: Use assign to avoid full DataFrame copy
        scaled_values = self.scaler.transform(data[self.variables])
        return data.assign(**{var: scaled_values[:, i] for i, var in enumerate(self.variables)})

    def descale(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Revert the scaling transformation on the specified variables.

        Args:
            data (pd.DataFrame): The DataFrame containing the scaled data.

        Returns:
            pd.DataFrame: A new DataFrame with the specified variables descaled.
        """
        # Optimize: Use assign to avoid full DataFrame copy
        descaled_values = self.scaler.inverse_transform(data[self.variables])
        return data.assign(**{var: descaled_values[:, i] for i, var in enumerate(self.variables)})
