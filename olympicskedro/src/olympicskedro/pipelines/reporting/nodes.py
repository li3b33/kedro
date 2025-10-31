import pandas as pd

def generate_classification_summary(classification: pd.DataFrame) -> pd.DataFrame:
    summary = classification.groupby("Model").agg(["mean", "std"]).round(3)
    formatted = summary.copy()
    for col in summary.columns.levels[0]:
        formatted[col] = summary[col].apply(lambda x: f"{x['mean']} ± {x['std']}", axis=1)
    return formatted

def generate_regression_summary(regression: pd.DataFrame) -> pd.DataFrame:
    summary = regression.groupby("Model").agg(["mean", "std"]).round(3)
    formatted = summary.copy()
    for col in summary.columns.levels[0]:
        formatted[col] = summary[col].apply(lambda x: f"{x['mean']} ± {x['std']}", axis=1)
    return formatted

def combine_final_summary(classification_summary: pd.DataFrame, regression_summary: pd.DataFrame) -> pd.DataFrame:
    classification_summary["tipo"] = "Clasificación"
    regression_summary["tipo"] = "Regresión"
    final = pd.concat([classification_summary, regression_summary])
    return final
