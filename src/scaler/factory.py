"""Factory pattern for creating scalers.

This module implements the factory pattern to create different types of
scalers based on scaler type specifications.
"""

from enum import Enum
from typing import List, Type, TYPE_CHECKING

from src.scaler.base_scaler import BaseScaler

if TYPE_CHECKING:
    from src.scaler.maxabs_scaler import MaxAbsScaler
    from src.scaler.minmax_scaler import MinMaxScaler
    from src.scaler.none_scaler import NoneScaler
    from src.scaler.robust_scaler import RobustScaler
    from src.scaler.standard_scaler import StandardScaler


class ScalerType(Enum):
    """Enumeration of available scaler types."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"

    @classmethod
    def from_str(cls, scaler_type_str: str) -> "ScalerType":
        """Convert a string to a ScalerType enum.

        Args:
            scaler_type_str (str): String representation of the scaler type.

        Returns:
            ScalerType: The corresponding ScalerType enum value.

        Raises:
            ValueError: If the string doesn't match any ScalerType.
        """
        scaler_type_str = scaler_type_str.lower()
        for scaler_type in cls:
            if scaler_type.value == scaler_type_str:
                return scaler_type
        raise ValueError(
            f"Invalid scaler type: {scaler_type_str}. "
            f"Valid options are: {', '.join([st.value for st in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available scaler types.

        Returns:
            list: List of scaler type strings.
        """
        return [scaler_type.value for scaler_type in cls]


class ScalerFactory:
    """Factory class for creating scalers.

    This class implements the factory pattern to instantiate the appropriate
    scaler based on the specified scaler type.
    """

    @staticmethod
    def _get_scaler_registry():
        """Lazy load the scaler registry to avoid circular imports."""
        from src.scaler.maxabs_scaler import MaxAbsScaler  # noqa: F811
        from src.scaler.minmax_scaler import MinMaxScaler  # noqa: F811
        from src.scaler.none_scaler import NoneScaler  # noqa: F811
        from src.scaler.robust_scaler import RobustScaler  # noqa: F811
        from src.scaler.standard_scaler import StandardScaler  # noqa: F811

        return {
            ScalerType.STANDARD: StandardScaler,
            ScalerType.MINMAX: MinMaxScaler,
            ScalerType.ROBUST: RobustScaler,
            ScalerType.MAXABS: MaxAbsScaler,
            ScalerType.NONE: NoneScaler,
        }

    @classmethod
    def create_scaler(cls, scaler_type: ScalerType, variables: List[str]) -> BaseScaler:
        """Create a scaler of the specified type.

        Args:
            scaler_type (ScalerType): The scaler type to use.
            variables (List[str]): List of variables to be scaled.

        Returns:
            BaseScaler: An instance of the requested scaler.

        Raises:
            ValueError: If the scaler type is not recognized.
        """
        scaler_registry = cls._get_scaler_registry()
        if scaler_type not in scaler_registry:
            raise ValueError(
                f"Unknown scaler type: {scaler_type}. "
                f"Available types: {', '.join([st.value for st in ScalerType])}"
            )

        scaler_class = scaler_registry[scaler_type]
        return scaler_class(variables=variables)

    @classmethod
    def get_scaler_class(cls, scaler_type: ScalerType) -> Type[BaseScaler]:
        """Get the scaler class for a given type.

        Args:
            scaler_type (ScalerType): The scaler type.

        Returns:
            Type[BaseScaler]: The scaler class.

        Raises:
            ValueError: If the scaler type is not recognized.
        """
        scaler_registry = cls._get_scaler_registry()
        if scaler_type not in scaler_registry:
            raise ValueError(
                f"Unknown scaler type: {scaler_type}. "
                f"Available types: {', '.join([st.value for st in ScalerType])}"
            )

        return scaler_registry[scaler_type]

    @classmethod
    def list_available_types(cls) -> list:
        """Get a list of all available scaler types.

        Returns:
            list: List of ScalerType values.
        """
        scaler_registry = cls._get_scaler_registry()
        return list(scaler_registry.keys())
