import math
from enum import Enum
from typing import List, Tuple, Union

import numpy as np

import pandas as pd

from config.constants import DATE_COLUMN, OBSERVATION_WINDOW, TRAIN_PERC


class DatasetDomain(Enum):
    """
    Enum for different dataset domains.

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
    """
    Enum for different INMET datasets.

    Attributes:
        BRASILIA_DF (tuple): Dataset for Brasília, DF with the filename 
            "A001_Brasilia_DF.csv" and columns ["P", "PrA", "T", "UR", "VV"].
        VITORIA_ES (tuple): Dataset for Vitoria, ES with the filename 
            "A612_Vitoria_ES.csv" and columns ["P", "PrA", "T", "UR", "VV"].
        PORTOALEGRE_RS (tuple): Dataset for Porto Alegre, RS with the filename 
            "A801_PortoAlegre_RS" and columns ["P", "PrA", "T", "UR", "VV"].
        SAOPAULO_SP (tuple): Dataset for São Paulo, SP with the filename 
            "A701_SAOPAULO_SP" and columns ["P", "PrA", "T", "UR", "VV"].
    """

    BRASILIA_DF = ("A001_Brasilia_DF.csv", ["P", "PrA", "T", "UR", "VV"])
    VITORIA_ES = ("A612_Vitoria_ES.csv", ["P", "PrA", "T", "UR", "VV"])
    PORTOALEGRE_RS = ("A801_PortoAlegre_RS.csv", ["P", "PrA", "T", "UR", "VV"])
    SAOPAULO_SP = ("A701_SAOPAULO_SP.csv", ["P", "PrA", "T", "UR", "VV"])


class UCIDatasets(Enum):
    """
    UCIDatasets is an enumeration of different UCI datasets, each represented by a tuple containing the filename and a list of relevant features.

    Attributes:
        AIR_QUALITY (tuple): Contains the filename "air_quality.csv" and a list of features ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"].
        PRSA_BEIJING (tuple): Contains the filename "prsa_beijing.csv" and a list of features ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"].
        APPLIANCES_ENERGY (tuple): Contains the filename "appliances_energy.csv" and a list of features ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"].
        METRO_TRAFFIC (tuple): Contains the filename "metro_traffic.csv" and a list of features ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"].
    """

    AIR_QUALITY = ("air_quality.csv", ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"])
    PRSA_BEIJING = ("prsa_beijing.csv", ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"])
    APPLIANCES_ENERGY = ("appliances_energy.csv", ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"])
    METRO_TRAFFIC = ("metro_traffic.csv", ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"])


class TCPDDatasets(Enum):
    """
    Enum for different TCPD datasets.

    Attributes:
        APPLE (tuple): Dataset for Apple stock with columns ["Close", "Volume"].
        BANK (tuple): Dataset for bank transactions with column ["Amount"].
        BEE_WAGGLE (tuple): Dataset for bee waggle dance with columns ["x", "y", "sin(theta)", "cos(theta)"].
        BITCOIN (tuple): Dataset for Bitcoin prices with column ["USD/Bitcoin"].
        BRENT_SPOT (tuple): Dataset for Brent spot prices with column ["Dollars/Barrel"].
        JFK_PASSENGERS (tuple): Dataset for JFK airport passengers with column ["Number of Passengers"].
        LGA_PASSENGERS (tuple): Dataset for LGA airport passengers with column ["Number of Passengers"].
        MEASLES (tuple): Dataset for measles cases with column ["V1"].
        OCCUPANCY (tuple): Dataset for occupancy with columns ["V1", "V2", "V3", "V4"].
        QUALITY_CONTROL_1 (tuple): Dataset for quality control with column ["V1"].
        QUALITY_CONTROL_2 (tuple): Dataset for quality control with column ["V1"].
        QUALITY_CONTROL_3 (tuple): Dataset for quality control with column ["V1"].
        QUALITY_CONTROL_4 (tuple): Dataset for quality control with column ["V1"].
        QUALITY_CONTROL_5 (tuple): Dataset for quality control with column ["V1"].
        RUN_LOG (tuple): Dataset for run logs with columns ["Pace", "Distance"].
        SCANLINE_42049 (tuple): Dataset for scanline with column ["Line 170"].
        SCANLINE_126007 (tuple): Dataset for scanline with column ["Line 200"].
        USD_ISK (tuple): Dataset for USD to ISK exchange rate with column ["Exchange rate"].
        US_POPULATION (tuple): Dataset for US population with column ["Population"].
        WELL_LOG (tuple): Dataset for well log with column ["V1"].
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
    """
    Enum class representing dummy datasets.

    Attributes:
        DUMMY (tuple): A tuple containing the filename of the dummy dataset and a list of column names.
    """

    DUMMY = ("dummy.csv", ["v1", "v2"])


def get_dataset_domain(domain: str) -> DatasetDomain:
    """
    Retrieves the corresponding DatasetDomain enum member for a given domain name.

    Args:
        domain (str): The name of the domain to retrieve.

    Returns:
        DatasetDomain: The corresponding DatasetDomain enum member.

    Raises:
        Exception: If no matching DatasetDomain is found for the given domain name.
    """
    for element in DatasetDomain:
        if element.name == domain:
            return element
    raise Exception("No DatasetDomain found for: " + domain)


def get_dataset(dataset: str) -> Union[INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]:
    """
    Retrieves a dataset object based on the provided dataset name.

    Args:
        dataset (str): The name of the dataset to retrieve.

    Returns:
        Union[INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]: The dataset object corresponding to the provided name.

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
    """
    Fills missing values in specified columns of a DataFrame using linear interpolation.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    variables (List[str]): A list of column names in the DataFrame where missing values should be filled.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled in the specified columns.
    """
    for variable in variables:
        df[variable] = df[variable].interpolate(method='linear')
    return df


def read_dataset(dataset_domain: Union[str, DatasetDomain],
                 dataset: Union[str, INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reads a dataset based on the provided domain and dataset identifiers.

    Args:
        dataset_domain (Union[str, DatasetDomain]): The domain of the dataset, either as a string or a DatasetDomain enum.
        dataset (Union[str, INMETDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]): The specific dataset to read, either as a string or an appropriate dataset enum.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the dataset as a pandas DataFrame and a list of variable names.

    Raises:
        ValueError: If the dataset domain or dataset is not recognized.

    Notes:
        - The dataset is read from a CSV file located in the "datasets/{dataset_domain}" directory.
        - The DataFrame is filtered to include only the date column and the specified variables.
        - If the dataset domain is INMET, missing values in the DataFrame are filled using the `fill_na` function.
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
    """
    Splits a DataFrame into training and testing sets based on a predefined training percentage.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training DataFrame and the testing DataFrame.
    """
    train_size = math.floor(df.shape[0] * TRAIN_PERC)
    train = df.iloc[:train_size].reset_index(drop=True)
    test = df.iloc[train_size:].reset_index(drop=True)

    return train, test


def split_X_y(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Splits the input DataFrame into features (X) and target (y) arrays for time series forecasting.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be split.

    Returns:
        Tuple[np.array, np.array]: A tuple containing two numpy arrays:
            - X: The feature array with shape (n_samples, OBSERVATION_WINDOW, n_features).
            - y: The target array with shape (n_samples, n_features).

    Notes:
        - The function assumes that the DataFrame contains a column named DATE_COLUMN which will be dropped if present.
        - The OBSERVATION_WINDOW constant defines the number of time steps to include in each sample.
    """
    X, y = [], []
    if DATE_COLUMN in df.columns:
        df = df.drop(columns=DATE_COLUMN)
    for i in range(len(df) - OBSERVATION_WINDOW):
        X.append(df.iloc[i:i + OBSERVATION_WINDOW])
        y.append(df.iloc[i + OBSERVATION_WINDOW])
    X = np.array(X)
    y = np.array(y)

    return X, y
