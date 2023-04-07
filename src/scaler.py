from typing import List

import pandas as pd

from sklearn import preprocessing as pp


class MaxAbsScaler:
    """The Scaler entity"""

    scaler = pp.MaxAbsScaler()

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the scaler
        :param data: the data to be fitted
        :return: the new dataframe
        """
        self.scaler.fit(data[self.variables])

    def fit_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and scales the data
        :param data: the data to be fitted and scaled
        :return: the new dataframe
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.fit_transform(data[self.variables])
        return data_output

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the data
        :param data: the data to be scaled
        :return: the new dataframe
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.transform(data_output[self.variables])
        return data_output

    def descale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Descales the data
        :param data: the data to be descaled
        :return: the new dataframe
        """
        data_output = data.copy()
        data_output[self.variables] = self.scaler.inverse_transform(data_output[self.variables])
        return data_output
