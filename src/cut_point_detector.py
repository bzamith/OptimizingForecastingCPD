import math
from typing import List, Union

import numpy as np

import pandas as pd

import ruptures as rpt

from config.constants import OBSERVATION_WINDOW


class CutPointDetector:
    def __init__(self):
        raise Exception("Not implemented")

    def get_stack(self, df: pd.DataFrame, variables: List[str]) -> np.vstack:
        stack_list = []
        for col in variables:
            stack_list.append(df[col].values)
        return np.vstack(stack_list).T

    def find_cut_point(self, df: pd.DataFrame, variables: List[str]) -> int:
        stacked_df = self.get_stack(df, variables)
        cut = self.model.fit_predict(stacked_df, n_bkps=1)
        return cut[0]

    def apply_cut_point(self, df: pd.DataFrame, cut_point: int) -> pd.DataFrame:
        assert cut_point < df.shape[0], f"Cut point {cut_point} out of dataframe range ({len(df)}"
        return df.iloc[cut_point:]


class WindowCutPointDetector(CutPointDetector):
    def __init__(self, method: Union[str, float]):
        self.method = method.lower()
        self.model = rpt.Window(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class BinSegCutPointDetector(CutPointDetector):
    def __init__(self, method: Union[str, float]):
        self.method = method.lower()
        self.model = rpt.Binseg(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class BottomUpCutPointDetector(CutPointDetector):
    def __init__(self, method: Union[str, float]):
        self.method = method.lower()
        self.model = rpt.BottomUp(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class FixedPercCutPointDetector(CutPointDetector):
    def __init__(self, method: Union[str, float]):
        assert isinstance(method, float), f"Expected float for method in Fixed Percentage, instead got {type(method)}"
        assert 0 <= method <= 1, f"Method {method} is out of range. Must be between 0 and 1."
        self.method = method

    def find_cut_point(self, df: pd.DataFrame, variables: List[str]) -> int:
        return math.floor(df.shape[0] * self.method)


def get_cut_point_detector(cut_point_model: str, cut_point_method: Union[str, float]) -> CutPointDetector:
    if cut_point_model.upper() == "WINDOW":
        return WindowCutPointDetector(cut_point_method)
    elif cut_point_model.upper() == "BIN_SEG":
        return BinSegCutPointDetector(cut_point_method)
    elif cut_point_model.upper() == "BOTTOM_UP":
        return BottomUpCutPointDetector(cut_point_method)
    elif cut_point_model.upper() == "FIXED_PERC":
        return FixedPercCutPointDetector(cut_point_method)
    else:
        raise Exception(f"Cut point model {cut_point_model} not implemented")
