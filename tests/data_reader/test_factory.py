"""Unit tests for data reader factory module."""

import pytest

from src.data_reader.factory import (
    AUTOFORMERDatasets,
    DataReaderFactory,
    DatasetDomain,
    DummyDatasets,
    INMETDatasets,
    TCPDDatasets,
    UCIDatasets,
)


class TestDatasetDomain:
    """Tests for DatasetDomain enum."""

    def test_from_str_valid_domains(self):
        """Test from_str with valid domain strings."""
        assert DatasetDomain.from_str("inmet") == DatasetDomain.INMET
        assert DatasetDomain.from_str("INMET") == DatasetDomain.INMET
        assert DatasetDomain.from_str("InMeT") == DatasetDomain.INMET
        assert DatasetDomain.from_str("autoformer") == DatasetDomain.AUTOFORMER
        assert DatasetDomain.from_str("AUTOFORMER") == DatasetDomain.AUTOFORMER
        assert DatasetDomain.from_str("uci") == DatasetDomain.UCI
        assert DatasetDomain.from_str("UCI") == DatasetDomain.UCI
        assert DatasetDomain.from_str("tcpd") == DatasetDomain.TCPD
        assert DatasetDomain.from_str("TCPD") == DatasetDomain.TCPD
        assert DatasetDomain.from_str("dummy") == DatasetDomain.DUMMY
        assert DatasetDomain.from_str("DUMMY") == DatasetDomain.DUMMY

    def test_from_str_invalid_domain(self):
        """Test from_str with invalid domain string raises error."""
        with pytest.raises(ValueError, match="Invalid dataset domain: invalid"):
            DatasetDomain.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            DatasetDomain.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all domain values."""
        available = DatasetDomain.list_available()

        assert len(available) == 5
        assert "inmet" in available
        assert "autoformer" in available
        assert "uci" in available
        assert "tcpd" in available
        assert "dummy" in available


class TestINMETDatasets:
    """Tests for INMETDatasets enum."""

    def test_from_str_valid_datasets(self):
        """Test from_str with valid INMET dataset strings."""
        assert INMETDatasets.from_str("brasilia_df") == INMETDatasets.BRASILIA_DF
        assert INMETDatasets.from_str("BRASILIA_DF") == INMETDatasets.BRASILIA_DF
        assert INMETDatasets.from_str("BrAsIlIa_Df") == INMETDatasets.BRASILIA_DF
        assert INMETDatasets.from_str("vitoria_es") == INMETDatasets.VITORIA_ES
        assert INMETDatasets.from_str("portoalegre_rs") == INMETDatasets.PORTOALEGRE_RS
        assert INMETDatasets.from_str("saopaulo_sp") == INMETDatasets.SAOPAULO_SP

    def test_from_str_invalid_dataset(self):
        """Test from_str with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Invalid INMET dataset: invalid"):
            INMETDatasets.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            INMETDatasets.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all dataset names."""
        available = INMETDatasets.list_available()

        assert len(available) == 4
        assert "BRASILIA_DF" in available
        assert "VITORIA_ES" in available
        assert "PORTOALEGRE_RS" in available
        assert "SAOPAULO_SP" in available

    def test_dataset_values(self):
        """Test that datasets contain correct filename and columns."""
        filename, columns = INMETDatasets.BRASILIA_DF.value
        assert filename == "A001_Brasilia_DF.csv"
        assert columns == ["P", "PrA", "T", "UR", "VV"]


class TestAUTOFORMERDatasets:
    """Tests for AUTOFORMERDatasets enum."""

    def test_from_str_valid_datasets(self):
        """Test from_str with valid AUTOFORMER dataset strings."""
        assert AUTOFORMERDatasets.from_str("weather") == AUTOFORMERDatasets.WEATHER
        assert AUTOFORMERDatasets.from_str("WEATHER") == AUTOFORMERDatasets.WEATHER
        assert AUTOFORMERDatasets.from_str("WeAtHeR") == AUTOFORMERDatasets.WEATHER

    def test_from_str_invalid_dataset(self):
        """Test from_str with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Invalid AUTOFORMER dataset: invalid"):
            AUTOFORMERDatasets.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            AUTOFORMERDatasets.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all dataset names."""
        available = AUTOFORMERDatasets.list_available()

        assert len(available) == 1
        assert "WEATHER" in available

    def test_dataset_values(self):
        """Test that datasets contain correct filename and columns."""
        filename, columns = AUTOFORMERDatasets.WEATHER.value
        assert filename == "weather.csv"
        assert len(columns) == 20  # Note: "rh (%)" and "VPmax (mbar)" are concatenated
        assert "p (mbar)" in columns
        assert "T (degC)" in columns


class TestUCIDatasets:
    """Tests for UCIDatasets enum."""

    def test_from_str_valid_datasets(self):
        """Test from_str with valid UCI dataset strings."""
        assert UCIDatasets.from_str("air_quality") == UCIDatasets.AIR_QUALITY
        assert UCIDatasets.from_str("AIR_QUALITY") == UCIDatasets.AIR_QUALITY
        assert UCIDatasets.from_str("AiR_QuAlItY") == UCIDatasets.AIR_QUALITY
        assert UCIDatasets.from_str("prsa_beijing") == UCIDatasets.PRSA_BEIJING
        assert UCIDatasets.from_str("appliances_energy") == UCIDatasets.APPLIANCES_ENERGY
        assert UCIDatasets.from_str("metro_traffic") == UCIDatasets.METRO_TRAFFIC

    def test_from_str_invalid_dataset(self):
        """Test from_str with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Invalid UCI dataset: invalid"):
            UCIDatasets.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            UCIDatasets.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all dataset names."""
        available = UCIDatasets.list_available()

        assert len(available) == 4
        assert "AIR_QUALITY" in available
        assert "PRSA_BEIJING" in available
        assert "APPLIANCES_ENERGY" in available
        assert "METRO_TRAFFIC" in available

    def test_dataset_values(self):
        """Test that datasets contain correct filename and columns."""
        filename, columns = UCIDatasets.AIR_QUALITY.value
        assert filename == "air_quality.csv"
        assert columns == ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"]


class TestTCPDDatasets:
    """Tests for TCPDDatasets enum."""

    def test_from_str_valid_datasets(self):
        """Test from_str with valid TCPD dataset strings."""
        assert TCPDDatasets.from_str("apple") == TCPDDatasets.APPLE
        assert TCPDDatasets.from_str("APPLE") == TCPDDatasets.APPLE
        assert TCPDDatasets.from_str("ApPlE") == TCPDDatasets.APPLE
        assert TCPDDatasets.from_str("bank") == TCPDDatasets.BANK
        assert TCPDDatasets.from_str("bee_waggle") == TCPDDatasets.BEE_WAGGLE
        assert TCPDDatasets.from_str("bitcoin") == TCPDDatasets.BITCOIN
        assert TCPDDatasets.from_str("brent_spot") == TCPDDatasets.BRENT_SPOT
        assert TCPDDatasets.from_str("jfk_passengers") == TCPDDatasets.JFK_PASSENGERS
        assert TCPDDatasets.from_str("lga_passengers") == TCPDDatasets.LGA_PASSENGERS
        assert TCPDDatasets.from_str("measles") == TCPDDatasets.MEASLES
        assert TCPDDatasets.from_str("occupancy") == TCPDDatasets.OCCUPANCY
        assert TCPDDatasets.from_str("quality_control_1") == TCPDDatasets.QUALITY_CONTROL_1
        assert TCPDDatasets.from_str("quality_control_2") == TCPDDatasets.QUALITY_CONTROL_2
        assert TCPDDatasets.from_str("quality_control_3") == TCPDDatasets.QUALITY_CONTROL_3
        assert TCPDDatasets.from_str("quality_control_4") == TCPDDatasets.QUALITY_CONTROL_4
        assert TCPDDatasets.from_str("quality_control_5") == TCPDDatasets.QUALITY_CONTROL_5
        assert TCPDDatasets.from_str("run_log") == TCPDDatasets.RUN_LOG
        assert TCPDDatasets.from_str("scanline_42049") == TCPDDatasets.SCANLINE_42049
        assert TCPDDatasets.from_str("scanline_126007") == TCPDDatasets.SCANLINE_126007
        assert TCPDDatasets.from_str("usd_isk") == TCPDDatasets.USD_ISK
        assert TCPDDatasets.from_str("us_population") == TCPDDatasets.US_POPULATION
        assert TCPDDatasets.from_str("well_log") == TCPDDatasets.WELL_LOG

    def test_from_str_invalid_dataset(self):
        """Test from_str with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Invalid TCPD dataset: invalid"):
            TCPDDatasets.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            TCPDDatasets.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all dataset names."""
        available = TCPDDatasets.list_available()

        assert len(available) == 20
        assert "APPLE" in available
        assert "BANK" in available
        assert "BEE_WAGGLE" in available
        assert "BITCOIN" in available
        assert "BRENT_SPOT" in available
        assert "JFK_PASSENGERS" in available
        assert "LGA_PASSENGERS" in available
        assert "MEASLES" in available
        assert "OCCUPANCY" in available
        assert "QUALITY_CONTROL_1" in available
        assert "QUALITY_CONTROL_2" in available
        assert "QUALITY_CONTROL_3" in available
        assert "QUALITY_CONTROL_4" in available
        assert "QUALITY_CONTROL_5" in available
        assert "RUN_LOG" in available
        assert "SCANLINE_42049" in available
        assert "SCANLINE_126007" in available
        assert "USD_ISK" in available
        assert "US_POPULATION" in available
        assert "WELL_LOG" in available

    def test_dataset_values(self):
        """Test that datasets contain correct filename and columns."""
        filename, columns = TCPDDatasets.APPLE.value
        assert filename == "apple.csv"
        assert columns == ["Close", "Volume"]


class TestDummyDatasets:
    """Tests for DummyDatasets enum."""

    def test_from_str_valid_datasets(self):
        """Test from_str with valid Dummy dataset strings."""
        assert DummyDatasets.from_str("dummy") == DummyDatasets.DUMMY
        assert DummyDatasets.from_str("DUMMY") == DummyDatasets.DUMMY
        assert DummyDatasets.from_str("DuMmY") == DummyDatasets.DUMMY

    def test_from_str_invalid_dataset(self):
        """Test from_str with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Invalid Dummy dataset: invalid"):
            DummyDatasets.from_str("invalid")

        with pytest.raises(ValueError, match="Valid options are"):
            DummyDatasets.from_str("nonexistent")

    def test_list_available(self):
        """Test list_available returns all dataset names."""
        available = DummyDatasets.list_available()

        assert len(available) == 1
        assert "DUMMY" in available

    def test_dataset_values(self):
        """Test that datasets contain correct filename and columns."""
        filename, columns = DummyDatasets.DUMMY.value
        assert filename == "dummy.csv"
        assert columns == ["v1", "v2"]


class TestDataReaderFactory:
    """Tests for DataReaderFactory class."""

    def test_get_dataset_inmet(self):
        """Test get_dataset returns correct INMET dataset."""
        dataset = DataReaderFactory.get_dataset(DatasetDomain.INMET, "brasilia_df")
        assert dataset == INMETDatasets.BRASILIA_DF

        dataset = DataReaderFactory.get_dataset(DatasetDomain.INMET, "SAOPAULO_SP")
        assert dataset == INMETDatasets.SAOPAULO_SP

    def test_get_dataset_autoformer(self):
        """Test get_dataset returns correct AUTOFORMER dataset."""
        dataset = DataReaderFactory.get_dataset(DatasetDomain.AUTOFORMER, "weather")
        assert dataset == AUTOFORMERDatasets.WEATHER

    def test_get_dataset_uci(self):
        """Test get_dataset returns correct UCI dataset."""
        dataset = DataReaderFactory.get_dataset(DatasetDomain.UCI, "air_quality")
        assert dataset == UCIDatasets.AIR_QUALITY

        dataset = DataReaderFactory.get_dataset(DatasetDomain.UCI, "METRO_TRAFFIC")
        assert dataset == UCIDatasets.METRO_TRAFFIC

    def test_get_dataset_tcpd(self):
        """Test get_dataset returns correct TCPD dataset."""
        dataset = DataReaderFactory.get_dataset(DatasetDomain.TCPD, "apple")
        assert dataset == TCPDDatasets.APPLE

        dataset = DataReaderFactory.get_dataset(DatasetDomain.TCPD, "BITCOIN")
        assert dataset == TCPDDatasets.BITCOIN

    def test_get_dataset_dummy(self):
        """Test get_dataset returns correct Dummy dataset."""
        dataset = DataReaderFactory.get_dataset(DatasetDomain.DUMMY, "dummy")
        assert dataset == DummyDatasets.DUMMY

    def test_get_dataset_invalid_domain(self):
        """Test get_dataset with invalid domain raises error."""

        class FakeDomain:
            value = "FAKE_DOMAIN"

        with pytest.raises(ValueError, match="Unknown dataset domain"):
            DataReaderFactory.get_dataset(FakeDomain, "any_dataset")

        with pytest.raises(ValueError, match="Available domains"):
            DataReaderFactory.get_dataset(FakeDomain, "any_dataset")

    def test_list_available(self):
        """Test list_available returns all domain keys."""
        available = DataReaderFactory.list_available()

        assert len(available) == 5
        assert DatasetDomain.INMET in available
        assert DatasetDomain.AUTOFORMER in available
        assert DatasetDomain.UCI in available
        assert DatasetDomain.TCPD in available
        assert DatasetDomain.DUMMY in available
