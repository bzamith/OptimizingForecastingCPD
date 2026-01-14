"""Factory pattern for creating data readers.

This module implements the factory pattern to create data readers for different
dataset domains and provides enumeration of available datasets.
"""

from enum import Enum
from typing import Union


class DatasetDomain(Enum):
    """Enumeration for different dataset domains.

    Attributes:
        INMET (str): Represents the INMET dataset domain.
        AUTOFORMER (str): Represents the AUTOFORMER dataset domain as in https://github.com/thuml/Autoformer/tree/main
        UCI (str): Represents the UCI dataset domain.
        TCPD (str): Represents the TCPD dataset domain.
        DUMMY (str): Represents a dummy dataset domain for testing purposes.
    """

    INMET = "inmet"
    AUTOFORMER = "autoformer"
    UCI = "uci"
    TCPD = "tcpd"
    DUMMY = "dummy"

    @classmethod
    def from_str(cls, domain_str: str) -> "DatasetDomain":
        """Convert a string to a DatasetDomain enum.

        Args:
            domain_str (str): String representation of the domain.

        Returns:
            DatasetDomain: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any domain.
        """
        domain_str = domain_str.lower()
        for domain in cls:
            if domain.value.lower() == domain_str:
                return domain
        raise ValueError(
            f"Invalid dataset domain: {domain_str}. "
            f"Valid options are: {', '.join([d.value for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available dataset domains.

        Returns:
            list: List of domain strings.
        """
        return [domain.value for domain in cls]


class INMETDatasets(Enum):
    """Enumeration for different INMET datasets.

    Attributes:
        BRASILIA_DF (tuple): Tuple with the filename "A001_Brasilia_DF.csv" and a list of columns
            ["P", "PrA", "T", "UR", "VV"] for Brasília, DF.
        VITORIA_ES (tuple): Tuple with the filename "A612_Vitoria_ES.csv" and a list of columns
            ["P", "PrA", "T", "UR", "VV"] for Vitoria, ES.
        PORTOALEGRE_RS (tuple): Tuple with the filename "A801_PortoAlegre_RS.csv" and a list of columns
            ["P", "PrA", "T", "UR", "VV"] for Porto Alegre, RS.
        SAOPAULO_SP (tuple): Tuple with the filename "A701_SAOPAULO_SP.csv" and a list of columns
            ["P", "PrA", "T", "UR", "VV"] for São Paulo, SP.
    """

    BRASILIA_DF = ("A001_Brasilia_DF.csv", ["P", "PrA", "T", "UR", "VV"])
    VITORIA_ES = ("A612_Vitoria_ES.csv", ["P", "PrA", "T", "UR", "VV"])
    PORTOALEGRE_RS = ("A801_PortoAlegre_RS.csv", ["P", "PrA", "T", "UR", "VV"])
    SAOPAULO_SP = ("A701_SaoPaulo_SP.csv", ["P", "PrA", "T", "UR", "VV"])

    @classmethod
    def from_str(cls, dataset_str: str) -> "INMETDatasets":
        """Convert a string to an INMETDatasets enum.

        Args:
            dataset_str (str): String representation of the dataset.

        Returns:
            INMETDatasets: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any dataset.
        """
        dataset_str = dataset_str.lower()
        for dataset in cls:
            if dataset.name.lower() == dataset_str:
                return dataset
        raise ValueError(
            f"Invalid INMET dataset: {dataset_str}. "
            f"Valid options are: {', '.join([d.name for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available INMET datasets.

        Returns:
            list: List of dataset names.
        """
        return [dataset.name for dataset in cls]


class AUTOFORMERDatasets(Enum):
    """Enumeration for different AUTOFORMER datasets.

    Attributes:
        WEATHER (tuple): Tuple with the filename "weather.csv" and a list of columns
            ["p (mbar)", "T (degC)", "rh (%)"
            "VPact (mbar)", "rho (g/m**3)", "wv (m/s)",
            "rain (mm)", "SWDR (W/m**2)"].
    """

    WEATHER = (
        "weather.csv",
        [
            "p (mbar)",
            "T (degC)",
            "rh (%)",
            "VPact (mbar)",
            "rho (g/m**3)",
            "wv (m/s)",
            "rain (mm)",
            "SWDR (W/m**2)",
        ],
    )

    @classmethod
    def from_str(cls, dataset_str: str) -> "AUTOFORMERDatasets":
        """Convert a string to an AUTOFORMERDatasets enum.

        Args:
            dataset_str (str): String representation of the dataset.

        Returns:
            AUTOFORMERDatasets: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any dataset.
        """
        dataset_str = dataset_str.lower()
        for dataset in cls:
            if dataset.name.lower() == dataset_str:
                return dataset
        raise ValueError(
            f"Invalid AUTOFORMER dataset: {dataset_str}. "
            f"Valid options are: {', '.join([d.name for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available AUTOFORMER datasets.

        Returns:
            list: List of dataset names.
        """
        return [dataset.name for dataset in cls]


class UCIDatasets(Enum):
    """Enumeration of different UCI datasets.

    Each dataset is represented by a tuple containing the filename and a list of relevant features.

    Attributes:
        AIR_QUALITY (tuple): Tuple with "air_quality.csv" and features
            ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"].
        PRSA_BEIJING (tuple): Tuple with "prsa_beijing.csv" and features
            ["pm2_5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"].
        APPLIANCES_ENERGY (tuple): Tuple with "appliances_energy.csv" and features
            ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"].
        METRO_TRAFFIC (tuple): Tuple with "metro_traffic.csv" and features
            ["temp", "rain_1h", "clouds_all"].
    """

    AIR_QUALITY = ("air_quality.csv", ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"])
    PRSA_BEIJING = ("prsa_beijing.csv", ["pm2_5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"])
    APPLIANCES_ENERGY = (
        "appliances_energy.csv",
        ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"],
    )
    METRO_TRAFFIC = ("metro_traffic.csv", ["temp", "rain_1h", "clouds_all"])

    @classmethod
    def from_str(cls, dataset_str: str) -> "UCIDatasets":
        """Convert a string to a UCIDatasets enum.

        Args:
            dataset_str (str): String representation of the dataset.

        Returns:
            UCIDatasets: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any dataset.
        """
        dataset_str = dataset_str.lower()
        for dataset in cls:
            if dataset.name.lower() == dataset_str:
                return dataset
        raise ValueError(
            f"Invalid UCI dataset: {dataset_str}. "
            f"Valid options are: {', '.join([d.name for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available UCI datasets.

        Returns:
            list: List of dataset names.
        """
        return [dataset.name for dataset in cls]


class TCPDDatasets(Enum):
    """Enumeration for different TCPD datasets.

    Attributes:
        APPLE (tuple): Tuple with "apple.csv" and columns ["Close", "Volume"].
        BANK (tuple): Tuple with "bank.csv" and column ["Amount"].
        BEE_WAGGLE (tuple): Tuple with "bee_waggle_6.csv" and columns ["x", "y", "sin(theta)", "cos(theta)"].
        BITCOIN (tuple): Tuple with "bitcoin.csv" and column ["USD/Bitcoin"].
        BRENT_SPOT (tuple): Tuple with "brent_spot.csv" and column ["Dollars/Barrel"].
        JFK_PASSENGERS (tuple): Tuple with "jfk_passengers.csv" and column ["Number of Passengers"].
        LGA_PASSENGERS (tuple): Tuple with "lga_passengers.csv" and column ["Number of Passengers"].
        MEASLES (tuple): Tuple with "measles.csv" and column ["V1"].
        OCCUPANCY (tuple): Tuple with "occupancy.csv" and columns ["V1", "V2", "V3", "V4"].
        QUALITY_CONTROL_1 (tuple): Tuple with "quality_control_1.csv" and column ["V1"].
        QUALITY_CONTROL_2 (tuple): Tuple with "quality_control_2.csv" and column ["V1"].
        QUALITY_CONTROL_3 (tuple): Tuple with "quality_control_3.csv" and column ["V1"].
        QUALITY_CONTROL_4 (tuple): Tuple with "quality_control_4.csv" and column ["V1"].
        QUALITY_CONTROL_5 (tuple): Tuple with "quality_control_5.csv" and column ["V1"].
        RUN_LOG (tuple): Tuple with "run_log.csv" and columns ["Pace", "Distance"].
        SCANLINE_42049 (tuple): Tuple with "scanline_42049.csv" and column ["Line 170"].
        SCANLINE_126007 (tuple): Tuple with "scanline_126007.csv" and column ["Line 200"].
        USD_ISK (tuple): Tuple with "usd_isk.csv" and column ["Exchange rate"].
        US_POPULATION (tuple): Tuple with "us_population.csv" and column ["Population"].
        WELL_LOG (tuple): Tuple with "well_log.csv" and column ["V1"].
    """

    APPLE = ("apple.csv", ["Close", "Volume"])
    BANK = ("bank.csv", ["Amount"])
    BEE_WAGGLE = ("bee_waggle_6.csv", ["x", "y", "sin(theta)", "cos(theta)"])
    BITCOIN = ("bitcoin.csv", ["USD/Bitcoin"])
    BRENT_SPOT = ("brent_spot.csv", ["Dollars/Barrel"])
    JFK_PASSENGERS = ("jfk_passengers.csv", ["Number of Passengers"])
    LGA_PASSENGERS = ("lga_passengers.csv", ["Number of Passengers"])
    MEASLES = ("measles.csv", ["V1"])
    OCCUPANCY = ("occupancy.csv", ["V1", "V2", "V3", "V4"])
    QUALITY_CONTROL_1 = ("quality_control_1.csv", ["V1"])
    QUALITY_CONTROL_2 = ("quality_control_2.csv", ["V1"])
    QUALITY_CONTROL_3 = ("quality_control_3.csv", ["V1"])
    QUALITY_CONTROL_4 = ("quality_control_4.csv", ["V1"])
    QUALITY_CONTROL_5 = ("quality_control_5.csv", ["V1"])
    RUN_LOG = ("run_log.csv", ["Pace", "Distance"])
    SCANLINE_42049 = ("scanline_42049.csv", ["Line 170"])
    SCANLINE_126007 = ("scanline_126007.csv", ["Line 200"])
    USD_ISK = ("usd_isk.csv", ["Exchange rate"])
    US_POPULATION = ("us_population.csv", ["Population"])
    WELL_LOG = ("well_log.csv", ["V1"])

    @classmethod
    def from_str(cls, dataset_str: str) -> "TCPDDatasets":
        """Convert a string to a TCPDDatasets enum.

        Args:
            dataset_str (str): String representation of the dataset.

        Returns:
            TCPDDatasets: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any dataset.
        """
        dataset_str = dataset_str.lower()
        for dataset in cls:
            if dataset.name.lower() == dataset_str:
                return dataset
        raise ValueError(
            f"Invalid TCPD dataset: {dataset_str}. "
            f"Valid options are: {', '.join([d.name for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available TCPD datasets.

        Returns:
            list: List of dataset names.
        """
        return [dataset.name for dataset in cls]


class DummyDatasets(Enum):
    """Enumeration representing dummy datasets.

    Attributes:
        DUMMY (tuple): Tuple containing the filename of the dummy dataset and a list of column names.
    """

    DUMMY = ("dummy.csv", ["v1", "v2"])

    @classmethod
    def from_str(cls, dataset_str: str) -> "DummyDatasets":
        """Convert a string to a DummyDatasets enum.

        Args:
            dataset_str (str): String representation of the dataset.

        Returns:
            DummyDatasets: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any dataset.
        """
        dataset_str = dataset_str.lower()
        for dataset in cls:
            if dataset.name.lower() == dataset_str:
                return dataset
        raise ValueError(
            f"Invalid Dummy dataset: {dataset_str}. "
            f"Valid options are: {', '.join([d.name for d in cls])}"
        )

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available Dummy datasets.

        Returns:
            list: List of dataset names.
        """
        return [dataset.name for dataset in cls]


class DataReaderFactory:
    """Factory class for getting dataset enums based on domain.

    This class implements the factory pattern to retrieve the appropriate
    dataset enum based on the specified domain.
    """

    _dataset_registry = {
        DatasetDomain.INMET: INMETDatasets,
        DatasetDomain.AUTOFORMER: AUTOFORMERDatasets,
        DatasetDomain.UCI: UCIDatasets,
        DatasetDomain.TCPD: TCPDDatasets,
        DatasetDomain.DUMMY: DummyDatasets,
    }

    @classmethod
    def get_dataset(
        cls, domain: DatasetDomain, dataset_str: str
    ) -> Union[INMETDatasets, AUTOFORMERDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]:
        """Get the appropriate dataset enum based on domain.

        Args:
            domain (DatasetDomain): The dataset domain.
            dataset_str (str): String name of the dataset.

        Returns:
            Union[INMETDatasets, AUTOFORMERDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]: The dataset enum.

        Raises:
            ValueError: If domain is not recognized.
        """
        dataset_enum_class = cls._dataset_registry.get(domain)
        if dataset_enum_class is None:
            raise ValueError(
                f"Unknown dataset domain: {domain}. "
                f"Available domains: {', '.join([d.value for d in DatasetDomain])}"
            )

        return dataset_enum_class.from_str(dataset_str)

    @classmethod
    def list_available(cls) -> list:
        """Get a list of all available dataset domains.

        Returns:
            list: List of DatasetDomain values.
        """
        return list(cls._dataset_registry.keys())
