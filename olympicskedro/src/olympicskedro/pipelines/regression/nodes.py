import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import GridSearchCV

def prepare_dictionary_data(df: pd.DataFrame):
    """
    Prepara X e y para regresión (predice GDP per Capita).
    Mantiene solo columnas numéricas y elimina nulos.
    """
    df = df.copy()
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop("GDP per Capita", axis=1)
    y = df["GDP per Capita"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_winter_data(df: pd.DataFrame):
    """
    Prepara los datos para regresión: elimina nulos y deja solo columnas numéricas.
    Se asume que la variable objetivo es 'gdp_per_capita'.
    """
    df = df.copy()
    df = df.select_dtypes(include=["number"]).dropna()

    # Asegurar que la columna objetivo exista
    if "gdp_per_capita" not in df.columns:
        raise ValueError(":x: No se encontró la columna 'gdp_per_capita' en los datos.")

    X = df.drop("gdp_per_capita", axis=1)
    y = df["gdp_per_capita"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_regression_models(data):
    """
    Entrena 5 modelos de regresión con GridSearchCV y guarda los .pkl en data/06_models/regression.
    Devuelve un DataFrame con métricas.
    """
    X_train, X_test, y_train, y_test = data

    models = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "ridge": {
            "model": Ridge(),
            "params": {"model__alpha": [0.1, 1, 10]}
        },
        "lasso": {
            "model": Lasso(),
            "params": {"model__alpha": [0.001, 0.01, 0.1, 1]}
        },
        "random_forest": {
            "model": RandomForestRegressor(),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1]
            }
        },
        "svr": {
            "model": SVR(),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"]
            }
        }
    }

    results = []
    os.makedirs("data/06_models/regression", exist_ok=True)

    for name, config in models.items():
        print(f":mag: Ejecutando GridSearchCV para {name}...")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", config["model"])
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Best_Params": grid.best_params_,
            "RMSE": rmse,
            "R2_Score": r2
        })

        joblib.dump(best_model, f"data/06_models/regression/{name}.pkl")

    return pd.DataFrame(results)
