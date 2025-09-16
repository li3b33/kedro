import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def process_winter(winter: pd.DataFrame) -> pd.DataFrame:
    # Ejemplo de limpieza básica
    winter = winter.dropna()
    return winter

def process_summer(summer: pd.DataFrame) -> pd.DataFrame:
    summer = summer.dropna()
    return summer

def process_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    dic = dictionary.copy()
    dic = dic.drop_duplicates()

    # Rellenar valores nulos con la mediana
    if "GDP per Capita" in dic.columns:
        dic["GDP per Capita"] = dic["GDP per Capita"].fillna(dic["GDP per Capita"].median())

        #truncar valores
        dic["GDP per Capita"] = dic["GDP per Capita"].fillna(dic["GDP per Capita"].median())
        dic["GDP per Capita"] = np.trunc(dic["GDP per Capita"]).astype(int)

    if "Population" in dic.columns:
        dic["Population"] = dic["Population"].fillna(dic["Population"].median())

    return dic

def scale_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    numeric_cols = dictionary.select_dtypes(include="number").columns
    dictionary[numeric_cols] = scaler.fit_transform(dictionary[numeric_cols])
    return dictionary

def add_medal_flag(summer: pd.DataFrame) -> pd.DataFrame:
    summer = summer.copy()
    summer["medal_flag"] = summer["Medal"].notna().astype(int)
    return summer