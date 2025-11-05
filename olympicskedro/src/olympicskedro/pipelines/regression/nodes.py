import os
import joblib
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

RANDOM_STATE = 42

def prepare_dictionary_data(df: pd.DataFrame):
    """
    Preparación de datos con feature engineering extensivo
    """
    df = df.copy()
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    
    # Verificar target
    if "gdp_per_capita" not in df.columns:
        raise ValueError("No se encontró 'gdp_per_capita' en el dataset.")
    
    # FEATURE ENGINEERING EXTENSIVO
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # 1. Crear características de interacción
    potential_interactions = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if col1 != "gdp_per_capita" and col2 != "gdp_per_capita":
                potential_interactions.append((col1, col2))
    
    # Agregar algunas interacciones (no todas para evitar sobrecarga)
    for col1, col2 in potential_interactions[:5]:  # Limitar a 5 interacciones
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # 2. Transformaciones polinomiales para features importantes
    important_looking_cols = [col for col in numeric_cols if col != "gdp_per_capita" and df[col].nunique() > 5]
    for col in important_looking_cols[:3]:  # Limitar a 3
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    # 3. Estadísticas agregadas
    if len(important_looking_cols) > 0:
        df['numeric_mean'] = df[important_looking_cols].mean(axis=1)
        df['numeric_std'] = df[important_looking_cols].std(axis=1)
    
    # Limpieza final
    numeric_cols_updated = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Eliminar columnas con muchos missing o poca varianza
    df_clean = df[numeric_cols_updated].copy()
    
    # Filtrar columnas con suficiente varianza
    variance_threshold = 0.01
    numeric_variance = df_clean.var()
    valid_columns = numeric_variance[numeric_variance > variance_threshold].index.tolist()
    
    # Asegurar que el target está incluido
    if "gdp_per_capita" not in valid_columns and "gdp_per_capita" in df_clean.columns:
        valid_columns.append("gdp_per_capita")
    
    df_clean = df_clean[valid_columns]
    
    # Manejar missing values
    df_clean = df_clean.fillna(df_clean.median())
    
    # Eliminar outliers extremos (más conservador)
    for column in df_clean.columns:
        if column != "gdp_per_capita":
            Q1 = df_clean[column].quantile(0.05)
            Q3 = df_clean[column].quantile(0.95)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[column] = np.clip(df_clean[column], lower_bound, upper_bound)
    
    # Separar features y target
    X = df_clean.drop("gdp_per_capita", axis=1, errors='ignore')
    y = df_clean["gdp_per_capita"]
    
    print(f"[prepare_dictionary_data] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[prepare_dictionary_data] Target stats - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"[prepare_dictionary_data] Target skew: {y.skew():.2f}")
    
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

def train_regression_models(data):
    X_train, X_test, y_train, y_test = data
    
    # VERIFICAR Y LIMPIAR DATOS
    print(f"[DEBUG] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[DEBUG] y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    # Limpiar datos
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Verificar consistencia de dimensiones
    if X_train.shape[1] != X_test.shape[1]:
        print(f"[WARNING] Diferente número de features: train={X_train.shape[1]}, test={X_test.shape[1]}")
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        print(f"[WARNING] Nuevo shape - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Estrategia de transformación del target
    target_skew = y_train.skew()
    print(f"[train_regression_models] Target skew: {target_skew:.2f}")
    
    if abs(target_skew) > 0.5:
        print("[train_regression_models] Aplicando log1p al target")
        y_train_trans = np.log1p(y_train)
        use_log_transform = True
    else:
        y_train_trans = y_train.copy()
        use_log_transform = False
    
    # Preprocessor más robusto
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
    ], remainder="drop")
    
    # Modelos más diversos
    estimators = {
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "ridge": Ridge(random_state=RANDOM_STATE),
        "lasso": Lasso(random_state=RANDOM_STATE, max_iter=2000)
    }
    
    if XGBRegressor is not None:
        estimators["xgboost"] = XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", verbosity=0)
    
    # Espacios de búsqueda más amplios
    param_distributions = {
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.3, 0.5, 0.7, 0.9]
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9]
        },
        "ridge": {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__solver": ['auto', 'svd', 'cholesky']
        },
        "lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10],
            "model__selection": ['cyclic', 'random']
        }
    }
    
    if XGBRegressor is not None:
        param_distributions["xgboost"] = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9]
        }
    
    os.makedirs("data/06_models/regression", exist_ok=True)
    results = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    
    for name, estimator in estimators.items():
        print(f"\n🔧 Entrenando {name} ...")
        
        try:
            # Pipeline con selección de características
            pipe = Pipeline([
                ("pre", preprocessor),
                ("feature_selector", SelectKBest(score_func=mutual_info_regression, 
                                               k=min(20, X_train.shape[1]))),
                ("model", estimator)
            ])
            
            n_iter = 15
            
            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_distributions[name],
                n_iter=n_iter,
                cv=5,  # CV más estable
                scoring="r2",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
            
            rs.fit(X_train, y_train_trans)
            
            best = rs.best_estimator_
            
            # Predecir y transformar inversamente si se usó log
            y_pred_test = best.predict(X_test)
            if use_log_transform:
                y_pred_test = np.expm1(y_pred_test)
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            mape = mean_absolute_percentage_error(y_test, y_pred_test)
            
            cv_scores = cross_val_score(best, X_train, y_train_trans, cv=5, scoring="r2", n_jobs=-1)
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            
            results.append({
                "Model": name,
                "Best_Params": str(rs.best_params_),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2_Score": float(r2),
                "MAPE": float(mape),
                "CV_R2_Mean": float(cv_mean),
                "CV_R2_Std": float(cv_std)
            })
            
            joblib.dump(best, f"data/06_models/regression/{name}.pkl")
            
            print(f"[{name}] R2_test={r2:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            print(f"[{name}] MAPE={mape:.4f}, CV_R2={cv_mean:.3f}±{cv_std:.3f}")
            
        except Exception as e:
            print(f"❌ Error entrenando {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(results)
    
    # ENSEMBLE FINAL con los mejores modelos
    print("\n🎯 Entrenando ensemble final...")
    try:
        # Seleccionar top 3 modelos por R2
        if len(results_df) >= 2:
            top_models = results_df.nlargest(2, 'R2_Score')['Model'].tolist()
            ensemble_estimators = []
            
            for model_name in top_models:
                model_path = f"data/06_models/regression/{model_name}.pkl"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    ensemble_estimators.append((model_name, model))
            
            if len(ensemble_estimators) >= 2:
                ensemble = VotingRegressor(estimators=ensemble_estimators)
                
                # Reentrenar el ensemble
                if use_log_transform:
                    ensemble.fit(X_train, y_train_trans)
                    y_pred_ensemble = ensemble.predict(X_test)
                    if use_log_transform:
                        y_pred_ensemble = np.expm1(y_pred_ensemble)
                else:
                    ensemble.fit(X_train, y_train)
                    y_pred_ensemble = ensemble.predict(X_test)
                
                # Métricas del ensemble
                rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
                mae_ens = mean_absolute_error(y_test, y_pred_ensemble)
                r2_ens = r2_score(y_test, y_pred_ensemble)
                mape_ens = mean_absolute_percentage_error(y_test, y_pred_ensemble)
                
                results_df = pd.concat([results_df, pd.DataFrame([{
                    "Model": "ENSEMBLE",
                    "Best_Params": "VotingRegressor",
                    "RMSE": float(rmse_ens),
                    "MAE": float(mae_ens),
                    "R2_Score": float(r2_ens),
                    "MAPE": float(mape_ens),
                    "CV_R2_Mean": float(r2_ens),
                    "CV_R2_Std": 0.0
                }])], ignore_index=True)
                
                joblib.dump(ensemble, "data/06_models/regression/ensemble.pkl")
                print(f"[ENSEMBLE] R2_test={r2_ens:.4f}, RMSE={rmse_ens:.3f}")
    
    except Exception as e:
        print(f"❌ Error creando ensemble: {str(e)}")

        
    # Guardar también métricas planas compatibles con DVC
    metrics_output = Path("data/08_reporting/regression_metrics_flat.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        row["Model"]: {
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "R2_Score": row["R2_Score"],
            "MAPE": row["MAPE"],
            "CV_R2_Mean": row["CV_R2_Mean"]
        }
        for _, row in results_df.iterrows()
    }

    # === FORMATEO DE MÉTRICAS PARA DVC ===
    # Redondea valores a 2 decimales y ordena las claves
    rounded_metrics = {
        model: {
            k: round(v, 2) if isinstance(v, (float, int)) else v
            for k, v in sorted(metrics.items())
        }
        for model, metrics in sorted(metrics_dict.items())
}

    # Guarda métricas en formato limpio para DVC
    with open(metrics_output, "w") as f:
        json.dump(rounded_metrics, f, indent=4, sort_keys=True)


    print(f"\n✅ Métricas guardadas en: {metrics_output}")


    # ==========================================================
    # MOSTRAR RESULTADOS FINALES
    # ==========================================================
    print("\n" + "="*60)
    print("RESULTADOS FINALES REGRESIÓN (ordenados por R2 Score):")
    print("="*60)
    print(results_df.sort_values("R2_Score", ascending=False).to_string(index=False))

    return results_df