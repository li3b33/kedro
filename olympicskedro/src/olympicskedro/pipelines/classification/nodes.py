import os
import joblib
import pandas as pd
from sklearn.svm import SVC
<<<<<<< HEAD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
=======
from sklearn.model_selection import GridSearchCV
>>>>>>> 68b05db74144302a901bcdd399921975b2fc6ddf

def prepare_summer_data(df: pd.DataFrame):
    """
    Convierte la columna 'Medal' en binaria:
    1 si ganó medalla de oro, 0 si fue plata o bronce.
    Mantiene solo columnas numéricas y elimina nulos.
    """
    df = df.copy()

    # Crear variable binaria: ganó oro o no
    df["Medal"] = (df["Medal"] == "Gold").astype(int)

    # Seleccionar solo columnas numéricas
    df = df.select_dtypes(include=["number"]).dropna()

    X = df.drop("Medal", axis=1)
    y = df["Medal"]

    # Verificar distribución de clases
    print("Distribución de clases:\n", y.value_counts())

    # División estratificada para balancear las clases
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_classification_models(data):
    X_train, X_test, y_train, y_test = data

    """
    Entrena 5 modelos de clasificación y guarda los .pkl en data/06_models/classification.
    Devuelve un DataFrame con métricas.
    """

    models = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=500, class_weight="balanced"),
            "params": {
                "model__C": [0.01, 0.1, 1, 10]
            }
        },
        "knn": {
            "model": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [3, 5, 7, 9]
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(class_weight="balanced"),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1]
            }
        },
        "svm": {
            "model": SVC(class_weight="balanced"),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"]
            }
        },
    }

    results = []
    os.makedirs("data/06_models/classification", exist_ok=True)

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
            scoring="f1",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Best_Params": grid.best_params_,
            "Accuracy": acc,
            "F1_Score": f1
        })

        joblib.dump(best_model, f"data/06_models/classification/{name}.pkl")

    return pd.DataFrame(results)
