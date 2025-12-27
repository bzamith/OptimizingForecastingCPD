"""Change Point Detection module with multiple detection methods.

This module provides a factory pattern for creating different types of
change point detectors including Window, Binary Segmentation, Bottom-Up, and Fixed Percentage.
"""

from src.cpd.base_cpd import BaseCPDDetector
from src.cpd.binseg_cpd import BinSegCPDDetector
from src.cpd.bottomup_cpd import BottomUpCPDDetector
from src.cpd.factory import CPDCostFunction, CPDDetectorFactory, CPDMethod
from src.cpd.fixedperc_cpd import FixedPercCPDDetector
from src.cpd.window_cpd import WindowCPDDetector

__all__ = [
    "BaseCPDDetector",
    "CPDMethod",
    "CPDCostFunction",
    "WindowCPDDetector",
    "BinSegCPDDetector",
    "BottomUpCPDDetector",
    "FixedPercCPDDetector",
    "CPDDetectorFactory",
]
