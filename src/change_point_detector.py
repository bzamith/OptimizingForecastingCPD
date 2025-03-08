import math
from enum import Enum
from typing import List, Tuple

import numpy as np

import pandas as pd

import ruptures as rpt

from config.constants import OBSERVATION_WINDOW


class ChangePointCostFunction(Enum):
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
    def from_str(cls, cost_function: str):
        for item in cls:
            if item.value.upper() == cost_function.upper() or item.name.upper() == cost_function.upper():
                return item
        raise ValueError(f"{cost_function} is not valid {cls.__name__}")


class ChangePointMethod(Enum):
    WINDOW = "Window"
    BIN_SEG = "Bin_Seg"
    BOTTOM_UP = "Bottom_Up"
    FIXED_PERC = "Fixed_Perc"

    @classmethod
    def from_str(cls, method: str):
        for item in cls:
            if item.value.upper() == method.upper() or item.name.upper() == method.upper():
                return item
        raise ValueError(f"{method} is not valid {cls.__name__}")


class ChangePointDetector:
    def __init__(self):
        raise Exception("Not implemented")

    def get_stack(self, df: pd.DataFrame, variables: List[str]) -> np.vstack:
        stack_list = []
        for col in variables:
            stack_list.append(df[col].values)
        return np.vstack(stack_list).T

    def find_change_point(self, df: pd.DataFrame, variables: List[str]) -> Tuple[int, float]:
        stacked_df = self.get_stack(df, variables)
        change_point = self.method.fit_predict(stacked_df, n_bkps=1)[0]
        change_point_perc = change_point * 100 / len(df)
        return change_point, change_point_perc

    def apply_change_point(self, df: pd.DataFrame, change_point: int) -> pd.DataFrame:
        assert change_point < df.shape[0], f"Cut point {change_point} out of dataframe range ({len(df)})"
        return df.iloc[change_point:]


class WindowChangePointDetector(ChangePointDetector):
    def __init__(self, cost_function: ChangePointCostFunction):
        self.cost_function = cost_function.value.lower()
        self.method = rpt.Window(model=self.cost_function, min_size=OBSERVATION_WINDOW + 1)


class BinSegChangePointDetector(ChangePointDetector):
    def __init__(self, cost_function: ChangePointCostFunction):
        self.cost_function = cost_function.value.lower()
        self.method = rpt.Binseg(model=self.cost_function, min_size=OBSERVATION_WINDOW + 1)


class BottomUpChangePointDetector(ChangePointDetector):
    def __init__(self, cost_function: ChangePointCostFunction):
        self.cost_function = cost_function.value.lower()
        self.method = rpt.BottomUp(model=self.cost_function, min_size=OBSERVATION_WINDOW + 1)


class FixedPercChangePointDetector(ChangePointDetector):
    def __init__(self, cost_function: ChangePointCostFunction):
        assert cost_function.value.startswith("Fixed_Cut"), f"Expected fixed cut for cost_function in Fixed Percentage, instead got {cost_function.value}"
        cost_function = float(cost_function.value[-3:])
        assert 0 <= cost_function <= 1, f"Fixed cut value {cost_function} is out of range. Must be between 0 and 1."
        self.cost_function = cost_function

    def find_change_point(self, df: pd.DataFrame, variables: List[str]) -> Tuple[int, float]:
        return math.floor(df.shape[0] * self.cost_function), self.cost_function


def get_change_point_detector(change_point_method: ChangePointMethod, change_point_cost_function: ChangePointCostFunction) -> ChangePointDetector:
    if change_point_method == ChangePointMethod.WINDOW:
        return WindowChangePointDetector(change_point_cost_function)
    elif change_point_method == ChangePointMethod.BIN_SEG:
        return BinSegChangePointDetector(change_point_cost_function)
    elif change_point_method == ChangePointMethod.BOTTOM_UP:
        return BottomUpChangePointDetector(change_point_cost_function)
    elif change_point_method == ChangePointMethod.FIXED_PERC:
        return FixedPercChangePointDetector(change_point_cost_function)
    else:
        raise Exception(f"Change point method {change_point_method.value} not implemented")
