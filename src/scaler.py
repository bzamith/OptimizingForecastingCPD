from typing import List

import pandas as pd

from sklearn import preprocessing as pp


class Scaler:
    """"
    Scaler class for scaling and descaling data using MinMaxScaler.

    Attributes:
    ----------
        scaler (MinMaxScaler): An instance of MinMaxScaler from sklearn.preprocessing.
        variables (List[str]): List of variables to be scaled.

    Methods:
     ----------
        fit(data: pd.DataFrame) -> None:
            Fits the scaler to the data.
            :param data: The data to be fitted.
        
        fit_scale(data: pd.DataFrame) -> pd.DataFrame:
            Fits and scales the data.
            :param data: The data to be fitted and scaled.
            :return: The new dataframe with scaled data.
        
        scale(data: pd.DataFrame) -> pd.DataFrame:
            Scales the data.
            :param data: The data to be scaled.
            :return: The new dataframe with scaled data.
        
        descale(data: pd.DataFrame) -> pd.DataFrame:
            Descales the data.
            :param data: The data to be descaled.
            :return: The new dataframe with descaled data.
    """

    scaler = pp.MinMaxScaler()

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the scaler to the provided data.

        Parameters:
        data (pd.DataFrame): The input data to fit the scaler on. It should contain the columns specified in self.variables.

        Returns:
        None
        """
        self.scaler.fit(data[self.variables])

    def fit_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler to the data and transforms the specified variables.

        Parameters:
        data (pd.DataFrame): The input data to be scaled.

        Returns:
        pd.DataFrame: The scaled data with the specified variables transformed.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.fit_transform(data[self.variables])
        return data_output

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the specified variables in the given DataFrame using the pre-fitted scaler.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be scaled.

        Returns:
        pd.DataFrame: A new DataFrame with the specified variables scaled.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.transform(data_output[self.variables])
        return data_output

    def descale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverts the scaling transformation applied to the specified variables in the DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the scaled data.

        Returns:
        pd.DataFrame: A new DataFrame with the specified variables descaled.
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.inverse_transform(data_output[self.variables])
        return data_output
