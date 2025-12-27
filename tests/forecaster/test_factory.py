"""Unit tests for forecaster factory and models."""

import pytest

from src.forecaster import (
    ARIMAForecasterHyperModel,
    ChronosForecasterHyperModel,
    ForecasterFactory,
    ForecasterType,
    HybridForecasterHyperModel,
    LSTMForecasterHyperModel,
    SSMForecasterHyperModel,
    TransformerForecasterHyperModel,
    TS2VECForecasterHyperModel,
)


class TestForecasterType:
    """Tests for ForecasterType enum."""

    def test_forecaster_type_from_str(self):
        """Test converting string to ForecasterType."""
        assert ForecasterType.from_str("LSTM") == ForecasterType.LSTM
        assert ForecasterType.from_str("Transformer") == ForecasterType.TRANSFORMER
        assert ForecasterType.from_str("SSM") == ForecasterType.SSM
        assert ForecasterType.from_str("Hybrid") == ForecasterType.HYBRID

    def test_forecaster_type_from_str_advanced(self):
        """Test converting advanced forecaster types."""
        assert ForecasterType.from_str("ARIMA") == ForecasterType.ARIMA
        assert ForecasterType.from_str("TS2Vec") == ForecasterType.TS2VEC
        assert ForecasterType.from_str("Chronos") == ForecasterType.CHRONOS

    def test_forecaster_type_from_str_invalid(self):
        """Test invalid forecaster type string."""
        with pytest.raises(ValueError, match="Invalid forecaster type"):
            ForecasterType.from_str("invalid_forecaster")

    def test_forecaster_type_from_str_case_sensitive(self):
        """Test that forecaster type is case-sensitive."""
        # This should fail because it's case-sensitive
        with pytest.raises(ValueError, match="Invalid forecaster type"):
            ForecasterType.from_str("lstm")

    def test_forecaster_type_list_available(self):
        """Test listing available forecaster types."""
        forecaster_types = ForecasterType.list_available()

        assert "LSTM" in forecaster_types
        assert "Transformer" in forecaster_types
        assert "SSM" in forecaster_types
        assert "Hybrid" in forecaster_types
        assert "ARIMA" in forecaster_types
        assert "TS2Vec" in forecaster_types
        assert "Chronos" in forecaster_types
        assert len(forecaster_types) == 7


class TestForecasterFactory:
    """Tests for ForecasterFactory."""

    def test_create_lstm_forecaster(self):
        """Test creating LSTM forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.LSTM, n_variables=3)

        assert isinstance(forecaster, LSTMForecasterHyperModel)
        assert forecaster.n_variables == 3

    def test_create_transformer_forecaster(self):
        """Test creating Transformer forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.TRANSFORMER, n_variables=2)

        assert isinstance(forecaster, TransformerForecasterHyperModel)
        assert forecaster.n_variables == 2

    def test_create_ssm_forecaster(self):
        """Test creating SSM forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.SSM, n_variables=1)

        assert isinstance(forecaster, SSMForecasterHyperModel)
        assert forecaster.n_variables == 1

    def test_create_hybrid_forecaster(self):
        """Test creating Hybrid forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.HYBRID, n_variables=4)

        assert isinstance(forecaster, HybridForecasterHyperModel)
        assert forecaster.n_variables == 4

    def test_create_arima_forecaster(self):
        """Test creating ARIMA forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.ARIMA, n_variables=1)

        assert isinstance(forecaster, ARIMAForecasterHyperModel)
        assert forecaster.n_variables == 1

    def test_create_ts2vec_forecaster(self):
        """Test creating TS2Vec forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.TS2VEC, n_variables=2)

        assert isinstance(forecaster, TS2VECForecasterHyperModel)
        assert forecaster.n_variables == 2

    def test_create_chronos_forecaster(self):
        """Test creating Chronos forecaster."""
        forecaster = ForecasterFactory.create_forecaster(ForecasterType.CHRONOS, n_variables=1)

        assert isinstance(forecaster, ChronosForecasterHyperModel)
        assert forecaster.n_variables == 1

    def test_create_forecaster_with_different_n_variables(self):
        """Test creating forecasters with different number of variables."""
        for n_vars in [1, 3, 5, 10]:
            forecaster = ForecasterFactory.create_forecaster(ForecasterType.LSTM, n_variables=n_vars)
            assert forecaster.n_variables == n_vars

    def test_get_model_class(self):
        """Test getting forecaster class."""
        model_class = ForecasterFactory.get_model_class(ForecasterType.LSTM)

        assert model_class == LSTMForecasterHyperModel

    def test_get_model_class_all_types(self):
        """Test getting all forecaster classes."""
        test_cases = [
            (ForecasterType.LSTM, LSTMForecasterHyperModel),
            (ForecasterType.TRANSFORMER, TransformerForecasterHyperModel),
            (ForecasterType.SSM, SSMForecasterHyperModel),
            (ForecasterType.HYBRID, HybridForecasterHyperModel),
            (ForecasterType.ARIMA, ARIMAForecasterHyperModel),
            (ForecasterType.TS2VEC, TS2VECForecasterHyperModel),
            (ForecasterType.CHRONOS, ChronosForecasterHyperModel),
        ]

        for forecaster_type, expected_class in test_cases:
            model_class = ForecasterFactory.get_model_class(forecaster_type)
            assert model_class == expected_class

    def test_list_available_models(self):
        """Test listing available forecaster models."""
        models = ForecasterFactory.list_available_models()

        assert ForecasterType.LSTM in models
        assert ForecasterType.TRANSFORMER in models
        assert ForecasterType.SSM in models
        assert ForecasterType.HYBRID in models
        assert ForecasterType.ARIMA in models
        assert ForecasterType.TS2VEC in models
        assert ForecasterType.CHRONOS in models
        assert len(models) == 7

    def test_create_forecaster_invalid_type(self):
        """Test creating forecaster with invalid type raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeType:
            value = "FAKE_FORECASTER"

        with pytest.raises(ValueError, match="Unknown forecaster type"):
            ForecasterFactory.create_forecaster(FakeType, n_variables=3)

    def test_get_model_class_invalid_type(self):
        """Test getting model class with invalid type raises error."""
        # Create a fake enum value that doesn't exist in registry
        class FakeType:
            value = "FAKE_FORECASTER"

        with pytest.raises(ValueError, match="Unknown forecaster type"):
            ForecasterFactory.get_model_class(FakeType)


class TestForecasterIntegration:
    """Integration tests for forecaster factory."""

    def test_factory_creates_correct_forecaster_types(self):
        """Test that factory creates correct forecaster instances."""
        test_cases = [
            (ForecasterType.LSTM, LSTMForecasterHyperModel),
            (ForecasterType.TRANSFORMER, TransformerForecasterHyperModel),
            (ForecasterType.SSM, SSMForecasterHyperModel),
            (ForecasterType.HYBRID, HybridForecasterHyperModel),
            (ForecasterType.ARIMA, ARIMAForecasterHyperModel),
            (ForecasterType.TS2VEC, TS2VECForecasterHyperModel),
            (ForecasterType.CHRONOS, ChronosForecasterHyperModel),
        ]

        for forecaster_type, expected_class in test_cases:
            forecaster = ForecasterFactory.create_forecaster(forecaster_type, n_variables=3)
            assert isinstance(forecaster, expected_class)
            assert forecaster.n_variables == 3

    def test_all_forecasters_have_required_attributes(self):
        """Test that all forecasters have required attributes."""
        for forecaster_type in ForecasterType:
            forecaster = ForecasterFactory.create_forecaster(forecaster_type, n_variables=2)

            # All forecasters should have n_variables attribute
            assert hasattr(forecaster, "n_variables")
            assert forecaster.n_variables == 2

            # All forecasters should have build method (from HyperModel)
            assert hasattr(forecaster, "build")
