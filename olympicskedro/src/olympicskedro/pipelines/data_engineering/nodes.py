import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    import pandas as pd
    winter = pd.read_parquet("data/01_raw/winter.csv")
    summer = pd.read_parquet("data/01_raw/summer.csv")
    dictionary = pd.read_parquet("data/01_raw/dictionary.csv")
    return winter, summer, dictionary

def clean_data(winter, summer, dictionary):
    """Limpiar datos: eliminar nulos y completar valores faltantes"""
    # Eliminar registros sin país
    winter = winter.dropna(subset=['Country'])
    summer = summer.dropna(subset=['Country'])
    
    # Rellenar datos socioeconómicos faltantes con mediana
    dictionary['GDP per Capita'] = dictionary['GDP per Capita'].fillna(
        dictionary['GDP per Capita'].median())
    dictionary['Population'] = dictionary['Population'].fillna(
        dictionary['Population'].median())
    
    return winter, summer, dictionary

def scale_features(dictionary):
    """Estandarizar variables numéricas"""
    scaler = StandardScaler()
    dictionary[['GDP_scaled', 'Population_scaled']] = scaler.fit_transform(
        dictionary[['GDP per Capita', 'Population']])
    return dictionary

def create_medal_flag(summer):
    """Crear variable bandera para medallas ganadas"""
    summer['Medal_Won'] = 1
    return summer

def save_processed_data(winter, summer, dictionary):
    """Guardar datos procesados en carpeta intermediate"""
    winter.to_parquet("data/02_intermediate/winter_procesed.parquet")
    summer.to_parquet("data/02_intermediate/summer_procesed.parquet")
    dictionary.to_parquet("data/02_intermediate/dictionary_procesed.parquet")
