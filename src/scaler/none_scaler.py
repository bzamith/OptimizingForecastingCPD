"""None Scaler implementation that performs no scaling."""

from typing import List

import pandas as pd

from src.scaler.base_scaler import BaseScaler


class NoneScaler(BaseScaler):
    """Pass-through scaler that performs no scaling transformation.

    This scaler is useful as a baseline or when no scaling is desired.
    All methods simply return a copy of the input data without modification.

    Attributes:
        variables (List[str]): List of variables (not used in this scaler).
    """

    def __init__(self, variables: List[str]):
        """Initialize the NoneScaler.

        Args:
            variables (List[str]): List of variables (not used).
        """
        super().__init__(variables)

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the scaler to the provided data (no-op).

        Args:
            data (pd.DataFrame): The input data (ignored).

        Returns:
            None
        """
        pass

    def fit_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data without scaling.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: A copy of the input DataFrame.
        """
        return data.copy()

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data without scaling.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A copy of the input DataFrame.
        """
        return data.copy()

    def descale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data without descaling.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.

        Returns:
            pd.DataFrame: A copy of the input DataFrame.
        """
        return data.copy()
