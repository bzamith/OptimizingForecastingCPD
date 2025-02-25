import math
from enum import Enum
from typing import List, Tuple, Union

import numpy as np

import pandas as pd

from config.constants import (
    DATE_COLUMN, FORECAST_HORIZON,
    OBSERVATION_WINDOW, TRAIN_PERC
)


class DatasetDomain(Enum):
    """Enumeration for different dataset domains.

    Attributes:
        INMET (str): Represents the INMET dataset domain.
        UCI (str): Represents the UCI dataset domain.
        TCPD (str): Represents the TCPD dataset domain.
        DUMMY (str): Represents a dummy dataset domain for testing purposes.
    """
    INMET = "inmet"
    UCI = "uci"
    TCPD = "tcpd"
    DUMMY = "dummy"


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
    SAOPAULO_SP = ("A701_SAOPAULO_SP.csv", ["P", "PrA", "T", "UR", "VV"])


class UCIDatasets(Enum):
    """Enumeration of different UCI datasets.

    Each dataset is represented by a tuple containing the filename and a list of relevant features.

    Attributes:
        AIR_QUALITY (tuple): Tuple with "air_quality.csv" and features
            ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"].
        PRSA_BEIJING (tuple): Tuple with "prsa_beijing.csv" and features
            ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"].
        APPLIANCES_ENERGY (tuple): Tuple with "appliances_energy.csv" and features
            ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"].
        METRO_TRAFFIC (tuple): Tuple with "metro_traffic.csv" and features
            ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"].
    """
    AIR_QUALITY = ("air_quality.csv", ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"])
    PRSA_BEIJING = ("prsa_beijing.csv", ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"])
    APPLIANCES_ENERGY = ("appliances_energy.csv", ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"])
    METRO_TRAFFIC = ("metro_traffic.csv", ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"])


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


class DummyDatasets(Enum):
    """Enumeration representing dummy datasets.

    Attributes:
        DUMMY (tuple): Tuple containing the filename of the dummy dataset and a list of column names.
    """
    DUMMY = ("dummy.csv", ["v1", "v2"])


def get_dataset_domain(domain: str) -> DatasetDomain:
    """Retrieve the DatasetDomain enum member for the given domain name.

    Args:
        domain (str): The name of the domain.

    Returns:
        DatasetDomain: The corresponding DatasetDomain enum member.

    Raises:
        Exception: If no matching DatasetDomain is found.
    """
    for element in DatasetDomain:
        if element.name == domain:
            return element
    raise Exception("No DatasetDomain found for: " + domain)


def get_dataset(dataset: str) -> Union[INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]:
    """Retrieve a dataset enum member based on the provided dataset name.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        Union[INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]: The corresponding dataset enum member.

    Raises:
        Exception: If no dataset is found with the provided name.
    """
    for element in INMETDatasets:
        if element.name == dataset:
            return element
    for element in UCIDatasets:
        if element.name == dataset:
            return element
    for element in TCPDDatasets:
        if element.name == dataset:
            return element
    for element in DummyDatasets:
        if element.name == dataset:
            return element
    raise Exception("No Dataset found for: " + dataset)


def fill_na(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Fill missing values in specified columns of the DataFrame using linear interpolation.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variables (List[str]): List of column names to fill missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in the specified columns.
    """
    for variable in variables:
        df[variable] = df[variable].interpolate(method='linear')
    return df


def read_dataset(
    dataset_domain: Union[str, DatasetDomain],
    dataset: Union[str, INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a dataset based on the provided domain and dataset identifiers.

    The CSV file is expected to be located in the "datasets/{dataset_domain}" directory.
    The resulting DataFrame is filtered to include the date column and the specified variables.
    For INMET domain datasets, missing values are filled using linear interpolation.

    Args:
        dataset_domain (Union[str, DatasetDomain]): The domain of the dataset.
        dataset (Union[str, INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]): The specific dataset to read.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame and a list of variable names.

    Raises:
        ValueError: If the dataset domain or dataset is not recognized.
    """
    if isinstance(dataset_domain, str):
        dataset_domain = get_dataset_domain(dataset_domain)
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)
    folder = f"datasets/{dataset_domain.value}"
    file = dataset.value[0]
    variables = dataset.value[1]

    df = pd.read_csv(f"{folder}/{file}")
    df = df[[DATE_COLUMN] + variables]

    if dataset_domain == DatasetDomain.INMET:
        df = fill_na(df, variables)

    return df, variables


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into training and testing sets based on a predefined training percentage.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and testing set DataFrames.
    """
    train_size = math.floor(df.shape[0] * TRAIN_PERC)
    train = df.iloc[:train_size].reset_index(drop=True)
    test = df.iloc[train_size:].reset_index(drop=True)
    return train, test


def split_X_y(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Split the DataFrame into feature and target arrays for time series forecasting.

    The function drops the DATE_COLUMN (if present) and generates samples based on OBSERVATION_WINDOW and FORECAST_HORIZON.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - X (np.array): Feature array with shape (n_samples, OBSERVATION_WINDOW, n_features).
            - y (np.array): Target array with shape (n_samples, n_features).

    Notes:
        - The OBSERVATION_WINDOW constant defines the number of time steps in each input sample.
        - The FORECAST_HORIZON constant defines the number of future time steps to predict.
    """
    X, y = [], []
    if DATE_COLUMN in df.columns:
        df = df.drop(columns=DATE_COLUMN)
    for i in range(len(df) - OBSERVATION_WINDOW - FORECAST_HORIZON + 1):
        X.append(df.iloc[i:i + OBSERVATION_WINDOW].values)
        y.append(df.iloc[i + OBSERVATION_WINDOW:i + OBSERVATION_WINDOW + FORECAST_HORIZON].values)
    X = np.array(X)
    y = np.array(y)
    return X, y
