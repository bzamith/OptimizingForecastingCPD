import math
from enum import Enum
from typing import List, Tuple, Union

import pandas as pd

DATE_COLUMN = "Date"
TRAIN_PERC = 0.8


class DatasetDomain(Enum):
    """Enum for different dataset domains"""

    EMBRAPA = "embrapa"
    INMET = "inmet"
    UCI = "uci"
    TCPD = "tcpd"


class EmbrapaDatasets(Enum):
    """Enum for different Embrapa datasets"""

    PANTANAL_CLIMATE = ("pantanal_climate.csv", ["T", "UR", "V", "P"])


class INMETDatasets(Enum):
    """Enum for different INMET datasets"""

    CRISTALINA_GO = ("A036_Cristalina_GO.csv", ["P", "PrA", "T", "UR", "VV"])
    BRASILIA_DF = ("A533_Brasilia_DF.csv", ["P", "PrA", "T", "UR", "VV"])
    IBIRITE_MG = ("A555_Ibirite_MG.csv", ["P", "PrA", "T", "UR", "VV"])


class UCIDatasets(Enum):
    """Enum for different UCI datasets"""

    AIR_QUALITY = ("air_quality.csv", ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"])
    PRSA_BEIJING = ("prsa_beijing.csv", ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"])
    APPLIANCES_ENERGY = ("appliances_energy.csv", ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"])
    METRO_TRAFFIC = ("metro_traffic.csv", ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"])


class TCPDDatasets(Enum):
    """Enum for different TCPD datasets"""

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


def get_dataset_domain(domain: str) -> DatasetDomain:
    for element in DatasetDomain:
        if element.name == domain:
            return element
    raise Exception("No DatasetDomain found for: " + domain)


def get_dataset(dataset: str) -> Union[EmbrapaDatasets, INMETDatasets, UCIDatasets, TCPDDatasets]:
    for element in EmbrapaDatasets:
        if element.name == dataset:
            return element
    for element in INMETDatasets:
        if element.name == dataset:
            return element
    for element in UCIDatasets:
        if element.name == dataset:
            return element
    for element in TCPDDatasets:
        if element.name == dataset:
            return element
    raise Exception("No Dataset found for: " + dataset)


def fill_na(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    for variable in variables:
        df[variable] = df[variable].interpolate(method='linear')
    return df


def read_dataset(dataset_domain: Union[str, DatasetDomain],
                 dataset: Union[str, EmbrapaDatasets, INMETDatasets, UCIDatasets, TCPDDatasets]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if isinstance(dataset_domain, str):
        dataset_domain = get_dataset_domain(dataset_domain)
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)
    folder = "datasets/" + dataset_domain.value + "/"
    file = dataset.value[0]
    variables = dataset.value[1]
    df = pd.read_csv(folder + file)
    df = df[[DATE_COLUMN] + variables]
    if dataset_domain == DatasetDomain.INMET:
        df = fill_na(df, variables)
    index_split = math.floor(df.shape[0] * TRAIN_PERC)
    X_train = df[:index_split].reset_index(drop=True)
    X_test = df[index_split:].reset_index(drop=True)
    return df, X_train, X_test, variables
