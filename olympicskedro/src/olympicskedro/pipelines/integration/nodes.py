import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import os
import json
from pathlib import Path

def integrate_clustering_features(summer_data: pd.DataFrame, clustering_results: pd.DataFrame):
    """
    Integra los resultados de clustering como features para modelos supervisados
    """
    print("Integrando características de clustering...")
    
    # Verificar y sincronizar índices
    print(f"Filas en summer_data: {len(summer_data)}")
    print(f"Filas en clustering_results: {len(clustering_results)}")
    
    # Resetear índices para asegurar alineación
    summer_data = summer_data.reset_index(drop=True)
    clustering_results = clustering_results.reset_index(drop=True)
    
    # Si hay discrepancia, tomar el mínimo común
    min_rows = min(len(summer_data), len(clustering_results))
    if len(summer_data) != len(clustering_results):
        print(f"⚠️  Discrepancia en número de filas. Usando {min_rows} filas comunes.")
        summer_data = summer_data.head(min_rows)
        clustering_results = clustering_results.head(min_rows)
    
    # Combinar datos originales con resultados de clustering
    integrated_data = summer_data.copy()
    
    # Agregar labels de clustering como features
    clustering_features = clustering_results[['kmeans_cluster', 'dbscan_cluster', 'hierarchical_cluster']].copy()
    
    # One-hot encoding para clusters categóricos
    kmeans_dummies = pd.get_dummies(clustering_features['kmeans_cluster'], prefix='kmeans')
    hierarchical_dummies = pd.get_dummies(clustering_features['hierarchical_cluster'], prefix='hierarchical')
    
    # Para DBSCAN, usar como feature numérica (incluye -1 para outliers)
    dbscan_feature = clustering_features['dbscan_cluster'].astype(int)
    
    # Combinar todas las features de clustering
    final_clustering_features = pd.concat([
        kmeans_dummies,
        hierarchical_dummies,
        dbscan_feature.rename('dbscan_cluster')
    ], axis=1)
    
    # Agregar features originales
    original_features = integrated_data.select_dtypes(include=["number"]).fillna(0)
    
    # Eliminar columnas problemáticas
    cols_to_remove = ['ID', 'Year', 'medal_gold']
    original_features = original_features[[col for col in original_features.columns if col not in cols_to_remove]]
    
    # Combinar features originales + clustering
    X_integrated = pd.concat([original_features, final_clustering_features], axis=1)
    
    # Target: ganar medalla de oro
    if "Medal" in integrated_data.columns:
        y = (integrated_data["Medal"] == "Gold").astype(int)
    elif "medal_gold" in integrated_data.columns:
        y = integrated_data["medal_gold"]
    else:
        # Crear target basado en cualquier medalla
        y = integrated_data["HasMedal"] if "HasMedal" in integrated_data.columns else pd.Series([0] * len(integrated_data))
    
    # Limpiar posibles valores NaN en el target
    y = y.fillna(0).astype(int)
    
    print(f"Dataset integrado: {X_integrated.shape}")
    print(f"Features de clustering añadidas: {final_clustering_features.shape[1]}")
    print(f"Balance de clases: {y.value_counts().to_dict()}")
    
    # Verificar que no hay NaN en los datos
    print(f"NaN en X_integrated: {X_integrated.isna().sum().sum()}")
    print(f"NaN en y: {y.isna().sum()}")
    
    return train_test_split(X_integrated, y, test_size=0.2, random_state=42, stratify=y)

def train_integrated_model(data):
    """
    Entrena modelo con features integradas (originales + clustering)
    """
    X_train, X_test, y_train, y_test = data
    
    print(f"Entrenando modelo integrado - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Balance en train: {y_train.value_counts().to_dict()}")
    print(f"Balance en test: {y_test.value_counts().to_dict()}")
    
    # Limpiar datos
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Preprocessor
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
    ], remainder="drop")
    
    # Modelos para comparación
    models = {
        "rf_integrated": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "gb_integrated": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    
    # Crear directorio para modelos integrados
    os.makedirs("data/06_models/integration", exist_ok=True)
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        
        try:
            pipe = Pipeline([
                ("pre", preprocessor),
                ("model", model)
            ])
            
            pipe.fit(X_train, y_train)
            
            # Predecir
            y_pred = pipe.predict(X_test)
            y_pred_proba = pipe.predict_proba(X_test)[:, 1]
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                "Model": name,
                "Accuracy": float(acc),
                "F1_Score": float(f1),
                "Precision": float(prec),
                "Recall": float(rec),
                "AUC_ROC": float(auc),
                "Features_Used": f"Original + Clustering ({X_train.shape[1]} features)"
            })
            
            # Guardar modelo
            joblib.dump(pipe, f"data/06_models/integration/{name}.pkl")
            print(f"✅ {name} entrenado y guardado - AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"❌ Error entrenando {name}: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("RESULTADOS MODELO INTEGRADO:")
    print("="*50)
    print(results_df.to_string(index=False))
    
    return results_df

def compare_performance(original_results: pd.DataFrame, integrated_results: pd.DataFrame):
    """
    Compara rendimiento entre modelos originales e integrados
    """
    comparison = []
    
    # Encontrar mejor modelo original (basado en AUC)
    if not original_results.empty and 'AUC_ROC' in original_results.columns:
        best_original = original_results.loc[original_results['AUC_ROC'].idxmax()]
        best_original_auc = best_original['AUC_ROC']
        best_original_model = best_original['Model']
    else:
        best_original_auc = 0.5  # Valor por defecto (random)
        best_original_model = "Baseline"
    
    # Encontrar mejor modelo integrado
    if not integrated_results.empty and 'AUC_ROC' in integrated_results.columns:
        best_integrated = integrated_results.loc[integrated_results['AUC_ROC'].idxmax()]
        best_integrated_auc = best_integrated['AUC_ROC']
        best_integrated_model = best_integrated['Model']
    else:
        best_integrated_auc = best_original_auc
        best_integrated_model = "No disponible"
    
    # Calcular mejora
    improvement = best_integrated_auc - best_original_auc
    improvement_pct = (improvement / best_original_auc) * 100 if best_original_auc > 0 else 0
    
    comparison_data = {
        'Best_Original_Model': best_original_model,
        'Best_Original_AUC': round(best_original_auc, 4),
        'Best_Integrated_Model': best_integrated_model,
        'Best_Integrated_AUC': round(best_integrated_auc, 4),
        'AUC_Improvement': round(improvement, 4),
        'Improvement_Percentage': round(improvement_pct, 2),
        'Integration_Successful': improvement > 0
    }
    
    comparison_df = pd.DataFrame([comparison_data])
    
    print("\n" + "="*60)
    print("COMPARACIÓN: ORIGINAL vs INTEGRADO")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    if improvement > 0:
        print(f"✅ MEJORA DETECTADA: +{improvement_pct:.2f}% en AUC")
    else:
        print("❌ No se detectó mejora con la integración")
    
    return comparison_df

def save_integration_metrics(comparison_results: pd.DataFrame):
    """
    Guarda métricas de integración para DVC
    """
    if not comparison_results.empty:
        metrics = comparison_results.iloc[0].to_dict()
        
        # Formatear métricas para DVC
        dvc_metrics = {
            "integration_improvement_pct": round(metrics.get('Improvement_Percentage', 0), 2),
            "integration_successful": metrics.get('Integration_Successful', False),
            "best_integrated_auc": round(metrics.get('Best_Integrated_AUC', 0), 4),
            "best_original_auc": round(metrics.get('Best_Original_AUC', 0), 4)
        }
        
        metrics_output = Path("data/08_reporting/integration_metrics.json")
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_output, "w") as f:
            json.dump(dvc_metrics, f, indent=4, sort_keys=True)
        
        print(f"✅ Métricas de integración guardadas en: {metrics_output}")
        
        return dvc_metrics
    return {}