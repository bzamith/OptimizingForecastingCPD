"""Factory pattern for creating change point detectors.

This module implements the factory pattern to create different types of
change point detectors based on method and cost function specifications.
"""

from enum import Enum
from typing import Type, TYPE_CHECKING

from src.cpd.base_cpd import BaseCPDDetector

if TYPE_CHECKING:
    from src.cpd.binseg_cpd import BinSegCPDDetector
    from src.cpd.bottomup_cpd import BottomUpCPDDetector
    from src.cpd.fixedperc_cpd import FixedPercCPDDetector
    from src.cpd.window_cpd import WindowCPDDetector


class CPDCostFunction(Enum):
    """Enumeration of available cost functions for change point detection."""

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
    FC9 = "Fixed_Cut_0.9"

    @classmethod
    def from_str(cls, cost_function_str: str) -> "CPDCostFunction":
        """Convert a string to a CPDCostFunction enum.

        Args:
            cost_function_str (str): String representation of the cost function.

        Returns:
            CPDCostFunction: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any cost function.
        """
        cost_function_str = cost_function_str.lower()
        for cost_function in cls:
            if cost_function.value.lower() == cost_function_str:
                return cost_function
        raise ValueError(
            f"Invalid cost function: {cost_function_str}. "
            f"Valid options are: {', '.join([cf.value for cf in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available cost functions.

        Returns:
            list: List of cost function strings.
        """
        return [cost_function.value for cost_function in cls]


class CPDMethod(Enum):
    """Enumeration of available change point detection methods."""

    WINDOW = "Window"
    BIN_SEG = "Bin_Seg"
    BOTTOM_UP = "Bottom_Up"
    FIXED_PERC = "Fixed_Perc"

    @classmethod
    def from_str(cls, method_str: str) -> "CPDMethod":
        """Convert a string to a CPDMethod enum.

        Args:
            method_str (str): String representation of the method.

        Returns:
            CPDMethod: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any method.
        """
        method_str = method_str.lower()
        for method in cls:
            if method.value.lower() == method_str:
                return method
        raise ValueError(
            f"Invalid change point method: {method_str}. "
            f"Valid options are: {', '.join([m.value for m in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available methods.

        Returns:
            list: List of method type strings.
        """
        return [method.value for method in cls]


class CPDDetectorFactory:
    """Factory class for creating change point detectors.

    This class implements the factory pattern to instantiate the appropriate
    change point detector based on the specified method and cost function.
    """

    @staticmethod
    def _get_detector_registry():
        """Lazy load the detector registry to avoid circular imports."""
        from src.cpd.binseg_cpd import BinSegCPDDetector  # noqa: F811
        from src.cpd.bottomup_cpd import BottomUpCPDDetector  # noqa: F811
        from src.cpd.fixedperc_cpd import FixedPercCPDDetector  # noqa: F811
        from src.cpd.window_cpd import WindowCPDDetector  # noqa: F811

        return {
            CPDMethod.WINDOW: WindowCPDDetector,
            CPDMethod.BIN_SEG: BinSegCPDDetector,
            CPDMethod.BOTTOM_UP: BottomUpCPDDetector,
            CPDMethod.FIXED_PERC: FixedPercCPDDetector,
        }

    @classmethod
    def create_detector(cls, method: CPDMethod, cost_function: CPDCostFunction) -> BaseCPDDetector:
        """Create a change point detector of the specified type.

        Args:
            method (CPDMethod): The detection method to use.
            cost_function (CPDCostFunction): The cost function to use.

        Returns:
            BaseCPDDetector: An instance of the requested detector.

        Raises:
            ValueError: If the method is not recognized.
        """
        detector_registry = cls._get_detector_registry()
        if method not in detector_registry:
            raise ValueError(
                f"Unknown change point method: {method}. "
                f"Available methods: {', '.join([m.value for m in CPDMethod])}"
            )

        detector_class = detector_registry[method]
        return detector_class(cost_function=cost_function)

    @classmethod
    def get_detector_class(cls, method: CPDMethod) -> Type[BaseCPDDetector]:
        """Get the detector class for a given method.

        Args:
            method (CPDMethod): The detection method.

        Returns:
            Type[BaseCPDDetector]: The detector class.

        Raises:
            ValueError: If the method is not recognized.
        """
        detector_registry = cls._get_detector_registry()
        if method not in detector_registry:
            raise ValueError(
                f"Unknown change point method: {method}. "
                f"Available methods: {', '.join([m.value for m in CPDMethod])}"
            )

        return detector_registry[method]

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available detection methods.

        Returns:
            list: List of CPDMethod values.
        """
        detector_registry = cls._get_detector_registry()
        return list(detector_registry.keys())
