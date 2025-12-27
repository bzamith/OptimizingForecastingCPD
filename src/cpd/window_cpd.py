"""Window-based change point detector implementation."""

from typing import List, Tuple

import pandas as pd
import ruptures as rpt

from config.constants import OBSERVATION_WINDOW
from src.cpd.base_cpd import BaseCPDDetector
from src.cpd.factory import CPDCostFunction


class WindowCPDDetector(BaseCPDDetector):
    """Window-based change point detector using sliding window approach.

    This detector uses a sliding window to scan through the time series
    and detect change points based on the specified cost function.

    Attributes:
        cost_function (CPDCostFunction): Cost function to use.
        method: Ruptures Window detector instance.
    """

    def __init__(self, cost_function: CPDCostFunction):
        """Initialize the Window change point detector.

        Args:
            cost_function (CPDCostFunction): Cost function for detection.
        """
        super().__init__(cost_function)
        cost_model = cost_function.value.lower()
        self.method = rpt.Window(model=cost_model, min_size=OBSERVATION_WINDOW + 1)

    def find_change_point(self, df: pd.DataFrame, variables: List[str]) -> Tuple[int, float]:
        """Find change point using sliding window approach.

        Args:
            df (pd.DataFrame): Time series data.
            variables (List[str]): Variables to analyze.

        Returns:
            Tuple[int, float]: Change point index and percentage of data.
        """
        stacked_df = self.get_stack(df, variables)
        change_point = self.method.fit_predict(stacked_df, n_bkps=1)[0]
        change_point_perc = change_point * 100 / len(df)
        return change_point, change_point_perc
