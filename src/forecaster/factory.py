"""Factory pattern for creating forecaster models.

This module implements the factory pattern to create different types of
forecasting models based on a forecaster type specification.
"""

from enum import Enum
from typing import Type, TYPE_CHECKING

from src.forecaster.base_forecaster import BaseForecasterHyperModel

if TYPE_CHECKING:
    from src.forecaster.arima_forecaster import ARIMAForecasterHyperModel
    from src.forecaster.gru_forecaster import GRUForecasterHyperModel
    from src.forecaster.lstm_forecaster import LSTMForecasterHyperModel
    from src.forecaster.tcn_forecaster import TCNForecasterHyperModel


class ForecasterType(Enum):
    """Enumeration of available forecasting forecaster types."""

    ARIMA = "ARIMA"
    GRU = "GRU"
    LSTM = "LSTM"
    TCN = "TCN"

    @classmethod
    def from_str(cls, forecaster_type_str: str) -> "ForecasterType":
        """Convert a string to a ForecasterType enum.

        Args:
            forecaster_type_str (str): String representation of the forecaster type.

        Returns:
            ForecasterType: The corresponding ForecasterType enum value.

        Raises:
            ValueError: If the string doesn't match any ForecasterType.
        """
        for forecaster_type in cls:
            if forecaster_type.value == forecaster_type_str:
                return forecaster_type
        raise ValueError(
            f"Invalid forecaster type: {forecaster_type_str}. "
            f"Valid options are: {', '.join([mt.value for mt in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available forecaster types.

        Returns:
            list: List of forecaster type strings.
        """
        return [forecaster_type.value for forecaster_type in cls]


class ForecasterFactory:
    """Factory class for creating forecaster models.

    This class implements the factory pattern to instantiate the appropriate
    forecaster hypermodel based on the specified forecaster type.
    """

    @staticmethod
    def _get_forecaster_registry():
        """Lazy load the forecaster registry to avoid circular imports."""
        from src.forecaster.arima_forecaster import ARIMAForecasterHyperModel  # noqa: F811
        from src.forecaster.gru_forecaster import GRUForecasterHyperModel  # noqa: F811
        from src.forecaster.lstm_forecaster import LSTMForecasterHyperModel  # noqa: F811
        from src.forecaster.tcn_forecaster import TCNForecasterHyperModel  # noqa: F811

        return {
            ForecasterType.ARIMA: ARIMAForecasterHyperModel,
            ForecasterType.GRU: GRUForecasterHyperModel,
            ForecasterType.LSTM: LSTMForecasterHyperModel,
            ForecasterType.TCN: TCNForecasterHyperModel,
        }

    @classmethod
    def create_forecaster(
        cls, forecaster_type: ForecasterType, n_variables: int
    ) -> BaseForecasterHyperModel:
        """Create a forecaster hypermodel of the specified type.

        Args:
            forecaster_type (ForecasterType): The type of model to create.
            n_variables (int): Number of variables in the time series data.

        Returns:
            BaseForecasterHyperModel: An instance of the requested forecaster hypermodel.

        Raises:
            ValueError: If the forecaster type is not recognized.
        """
        forecaster_registry = cls._get_forecaster_registry()
        if forecaster_type not in forecaster_registry:
            raise ValueError(
                f"Unknown forecaster type: {forecaster_type}. "
                f"Available types: {', '.join([mt.value for mt in ForecasterType])}"
            )

        forecaster_class = forecaster_registry[forecaster_type]
        return forecaster_class(n_variables=n_variables)

    @classmethod
    def get_model_class(cls, forecaster_type: ForecasterType) -> Type[BaseForecasterHyperModel]:
        """Get the forecaster class for a given forecaster type.

        Args:
            forecaster_type (ForecasterType): The type of model.

        Returns:
            Type[BaseForecasterHyperModel]: The forecaster class.

        Raises:
            ValueError: If the forecaster type is not recognized.
        """
        forecaster_registry = cls._get_forecaster_registry()
        if forecaster_type not in forecaster_registry:
            raise ValueError(
                f"Unknown forecaster type: {forecaster_type}. "
                f"Available types: {', '.join([mt.value for mt in ForecasterType])}"
            )

        return forecaster_registry[forecaster_type]

    @classmethod
    def list_available_models(cls) -> list:
        """Get a list of all available forecaster types.

        Returns:
            list: List of ForecasterType values.
        """
        forecaster_registry = cls._get_forecaster_registry()
        return list(forecaster_registry.keys())
