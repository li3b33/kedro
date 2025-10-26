import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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


def train_regression_models(data):
    """
    Entrena 5 modelos de regresión y guarda los .pkl en data/06_models/regression.
    Devuelve un DataFrame con métricas.
    """
    X_train, X_test, y_train, y_test = data

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor()
    }

    results = []
    os.makedirs("data/06_models/regression", exist_ok=True)

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": name, "MSE": mse, "R2_Score": r2})

        # 💾 Guardar modelo entrenado
        joblib.dump(pipe, f"data/06_models/regression/{name}.pkl")

    return pd.DataFrame(results)
