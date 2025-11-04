import pandas as pd

def generate_classification_summary(classification: pd.DataFrame) -> pd.DataFrame:
    # Filtrar solo las columnas numéricas para evitar errores de tipo
    numeric_cols = classification.select_dtypes(include=["number"]).columns
    non_numeric_cols = [c for c in classification.columns if c not in numeric_cols]

    if non_numeric_cols:
        print(f"⚠️ Columnas no numéricas ignoradas en el resumen de clasificación: {non_numeric_cols}")

    # Agrupar solo por columnas numéricas
    summary = classification.groupby("Model")[numeric_cols].agg(["mean", "std"]).round(3)

    # Formatear el resultado para mostrar mean ± std
    formatted = summary.copy()
    for col in summary.columns.levels[0]:
        formatted[col] = summary[col].apply(lambda x: f"{x['mean']} ± {x['std']}", axis=1)

    return formatted


def generate_regression_summary(regression: pd.DataFrame) -> pd.DataFrame:
    # Igual que arriba, pero para regresión
    numeric_cols = regression.select_dtypes(include=["number"]).columns
    non_numeric_cols = [c for c in regression.columns if c not in numeric_cols]

    if non_numeric_cols:
        print(f"⚠️ Columnas no numéricas ignoradas en el resumen de regresión: {non_numeric_cols}")

    summary = regression.groupby("Model")[numeric_cols].agg(["mean", "std"]).round(3)

    formatted = summary.copy()
    for col in summary.columns.levels[0]:
        formatted[col] = summary[col].apply(lambda x: f"{x['mean']} ± {x['std']}", axis=1)

    return formatted


def combine_final_summary(classification_summary: pd.DataFrame, regression_summary: pd.DataFrame) -> pd.DataFrame:
    # Combinar los resultados en una sola tabla
    classification_summary["Tipo"] = "Clasificación"
    regression_summary["Tipo"] = "Regresión"
    final = pd.concat([classification_summary, regression_summary])
    return final

