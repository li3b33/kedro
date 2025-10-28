import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


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
    """
    Entrena 5 modelos de clasificación y guarda los .pkl en data/06_models/classification.
    Devuelve un DataFrame con métricas.
    """
    X_train, X_test, y_train, y_test = data

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC()
    }

    results = []
    os.makedirs("data/06_models/classification", exist_ok=True)

    for name, model in models.items():
        # Saltar si el set de entrenamiento solo tiene una clase
        if len(set(y_train)) < 2:
            print(f"⚠️ No se puede entrenar {name}: solo hay una clase en el set de entrenamiento.")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({"Model": name, "Accuracy": acc, "F1_Score": f1})
        print(f"✅ {name} entrenado. Accuracy: {acc:.3f}, F1: {f1:.3f}")

        # 💾 Guardar modelo entrenado
        joblib.dump(pipe, f"data/06_models/classification/{name}.pkl")

    return pd.DataFrame(results)
