import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y nulos generales"""
    df = df.drop_duplicates().dropna(how="all")
    return df


def truncate_gdp(df: pd.DataFrame) -> pd.DataFrame:
    """Trunca 'GDP per Capita' a 2 decimales"""
    if "GDP per Capita" in df.columns:
        df["GDP per Capita"] = np.trunc(df["GDP per Capita"] * 100) / 100
    return df


def fill_missing_gdp(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena valores NaN en 'GDP per Capita' con la media"""
    if "GDP per Capita" in df.columns:
        df["GDP per Capita"] = df["GDP per Capita"].fillna(df["GDP per Capita"].mean())
    return df


def scale_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Escala todas las columnas numéricas entre 0 y 1"""
    num_df = df.select_dtypes(include=["number"])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(num_df)
    scaled_df = pd.DataFrame(scaled, columns=num_df.columns)
    return scaled_df


def add_medal_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega una columna binaria indicando si ganó medalla"""
    df = df.copy()
    df["HasMedal"] = df["Medal"].notna().astype(int)
    return df
