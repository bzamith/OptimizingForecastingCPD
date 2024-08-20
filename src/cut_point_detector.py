import math
from enum import Enum
from typing import List, Tuple

import numpy as np

import pandas as pd

import ruptures as rpt

from config.constants import OBSERVATION_WINDOW


class CutPointMethod(Enum):
    L1 = "L1"
    L2 = "L2"
    NORMAL = "Normal"
    RBF = "RBF"
    COSINE = "Cosine"
    LINEAR = "Linear"
    CLINEAR = "Clinear"
    RANK = "Rank"
    MAHALANOBIS = "Mahalanobis"
    AR = "AR"
    FC0 = "Fixed_Cut_0.0"
    FC1 = "Fixed_Cut_0.1"
    FC2 = "Fixed_Cut_0.2"
    FC3 = "Fixed_Cut_0.3"
    FC4 = "Fixed_Cut_0.4"
    FC5 = "Fixed_Cut_0.5"
    FC6 = "Fixed_Cut_0.6"
    FC7 = "Fixed_Cut_0.7"
    FC8 = "Fixed_Cut_0.8"

    @classmethod
    def from_str(cls, method: str):
        for item in cls:
            if item.value.upper() == method.upper() or item.name.upper() == method.upper():
                return item
        raise ValueError(f"{method} is not valid {cls.__name__}")


class CutPointModel(Enum):
    WINDOW = "Window"
    BIN_SEG = "Bin_Seg"
    BOTTOM_UP = "Bottom_Up"
    FIXED_PERC = "Fixed_Perc"

    @classmethod
    def from_str(cls, model: str):
        for item in cls:
            if item.value.upper() == model.upper() or item.name.upper() == model.upper():
                return item
        raise ValueError(f"{model} is not valid {cls.__name__}")


class CutPointDetector:
    def __init__(self):
        raise Exception("Not implemented")

    def get_stack(self, df: pd.DataFrame, variables: List[str]) -> np.vstack:
        stack_list = []
        for col in variables:
            stack_list.append(df[col].values)
        return np.vstack(stack_list).T

    def find_cut_point(self, df: pd.DataFrame, variables: List[str]) -> Tuple[int, float]:
        stacked_df = self.get_stack(df, variables)
        cut = self.model.fit_predict(stacked_df, n_bkps=1)[0]
        cut_perc = cut * 100 / len(df)
        return cut, cut_perc

    def apply_cut_point(self, df: pd.DataFrame, cut_point: int) -> pd.DataFrame:
        assert cut_point < df.shape[0], f"Cut point {cut_point} out of dataframe range ({len(df)})"
        return df.iloc[cut_point:]


class WindowCutPointDetector(CutPointDetector):
    def __init__(self, method: CutPointMethod):
        self.method = method.value.lower()
        self.model = rpt.Window(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class BinSegCutPointDetector(CutPointDetector):
    def __init__(self, method: CutPointMethod):
        self.method = method.value.lower()
        self.model = rpt.Binseg(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class BottomUpCutPointDetector(CutPointDetector):
    def __init__(self, method: CutPointMethod):
        self.method = method.value.lower()
        self.model = rpt.BottomUp(model=self.method, min_size=OBSERVATION_WINDOW + 1)


class FixedPercCutPointDetector(CutPointDetector):
    def __init__(self, method: CutPointMethod):
        assert method.value.startswith("Fixed_Cut"), f"Expected fixed cut for method in Fixed Percentage, instead got {method.value}"
        method = float(method.value[-3:])
        assert 0 <= method <= 1, f"Fixed cut value {method} is out of range. Must be between 0 and 1."
        self.method = method

    def find_cut_point(self, df: pd.DataFrame, variables: List[str]) -> Tuple[int, float]:
        return math.floor(df.shape[0] * self.method), self.method


def get_cut_point_detector(cut_point_model: CutPointModel, cut_point_method: CutPointMethod) -> CutPointDetector:
    if cut_point_model == CutPointModel.WINDOW:
        return WindowCutPointDetector(cut_point_method)
    elif cut_point_model == CutPointModel.BIN_SEG:
        return BinSegCutPointDetector(cut_point_method)
    elif cut_point_model == CutPointModel.BOTTOM_UP:
        return BottomUpCutPointDetector(cut_point_method)
    elif cut_point_model == CutPointModel.FIXED_PERC:
        return FixedPercCutPointDetector(cut_point_method)
    else:
        raise Exception(f"Cut point model {cut_point_model.value} not implemented")
