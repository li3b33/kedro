import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

RANDOM_STATE = 42

def prepare_summer_data(df: pd.DataFrame):
    df = df.copy()
    
    # Verificar que existe la columna Medal
    if "Medal" not in df.columns:
        raise ValueError("No se encontró columna 'Medal' en datos")
    
    # Crear target binario
    df["medal_gold"] = (df["Medal"] == "Gold").astype(int)
    
    # ENGINEERING DE FEATURES CRÍTICOS
    # 1. Características temporales
    if 'Year' in df.columns:
        df['years_since_first'] = df['Year'] - df['Year'].min()
    
    # 2. Características de experiencia del atleta
    if 'Age' in df.columns:
        df['age_group'] = pd.cut(df['Age'], bins=[0, 20, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4])
    
    # 3. Estadísticas por país y deporte
    if all(col in df.columns for col in ['NOC', 'Sport']):
        country_stats = df.groupby('NOC')['medal_gold'].agg(['mean', 'count']).rename(columns={'mean': 'country_win_rate', 'count': 'country_participations'})
        sport_stats = df.groupby('Sport')['medal_gold'].agg(['mean', 'count']).rename(columns={'mean': 'sport_win_rate', 'count': 'sport_participations'})
        
        df = df.merge(country_stats, on='NOC', how='left')
        df = df.merge(sport_stats, on='Sport', how='left')
    
    # 4. One-hot encoding para variables categóricas importantes
    categorical_cols = ['Season', 'Sport', 'NOC', 'City']
    for col in categorical_cols:
        if col in df.columns and df[col].nunique() < 50:  # Evitar alta dimensionalidad
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
    
    # Seleccionar características numéricas
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Eliminar columnas con poca varianza o que son el target
    cols_to_remove = ['medal_gold', 'ID', 'Year']  # Ajustar según tu dataset
    features = [col for col in numeric_cols if col not in cols_to_remove and df[col].nunique() > 1]
    
    X = df[features].copy()
    y = df["medal_gold"]
    
    # Manejar valores NaN
    X = X.fillna(X.median())
    
    print(f"[prepare_summer_data] Shape: {X.shape}")
    print(f"[prepare_summer_data] Class balance:\n{y.value_counts()}")
    print(f"[prepare_summer_data] Positive class ratio: {y.mean():.4f}")
    
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

def train_classification_models(data):
    X_train, X_test, y_train, y_test = data
    
    # VERIFICAR Y LIMPIAR DATOS ANTES DE ENTRENAR
    print(f"[DEBUG] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[DEBUG] y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    # Verificar que no hay NaN o infinitos
    print(f"[DEBUG] NaN en X_train: {X_train.isna().sum().sum()}")
    print(f"[DEBUG] NaN en X_test: {X_test.isna().sum().sum()}")
    
    # Limpiar datos
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Verificar consistencia de dimensiones
    if X_train.shape[1] != X_test.shape[1]:
        print(f"[WARNING] Diferente número de features: train={X_train.shape[1]}, test={X_test.shape[1]}")
        # Mantener solo columnas comunes
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        print(f"[WARNING] Nuevo shape - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # ESTRATEGIA PARA DATOS DESBALANCEADOS
    positive_ratio = y_train.mean()
    print(f"[train_classification_models] Positive class ratio in training: {positive_ratio:.4f}")
    
    if positive_ratio < 0.1:  # Muy desbalanceado
        print("[train_classification_models] Usando SMOTEENN para datos muy desbalanceados")
        smote_enn = SMOTEENN(
            smote=SMOTE(
                sampling_strategy=0.3,
                random_state=RANDOM_STATE,
                k_neighbors=min(3, max(1, len(y_train[y_train==1])-1))
            ),
            random_state=RANDOM_STATE
        )
        sampling_method = smote_enn
    else:
        print("[train_classification_models] Usando SMOTE estándar")
        sampling_method = SMOTE(random_state=RANDOM_STATE)
    
    # Preprocessor mejorado
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ]), numeric_features),
    ], remainder="drop")
    
    # Modelos enfocados en datos desbalanceados - AHORA 5 MODELOS
    estimators = {
        "random_forest": RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "svm": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }
    
    if XGBClassifier is not None:
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        estimators["xgboost"] = XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss", 
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight
        )
    
    # Espacios de búsqueda optimizados para AUC/Precision - ACTUALIZADO CON LOS 2 NUEVOS MODELOS
    param_distributions = {
        "random_forest": {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [5, 8, 10, 12],
            "model__min_samples_split": [10, 20, 30],
            "model__min_samples_leaf": [5, 10, 15],
            "model__max_features": [0.3, 0.5, 0.7]
        },
        "logistic": {
            "model__C": [0.001, 0.01, 0.1, 1],
            "model__penalty": ['l1', 'l2'],
            "model__solver": ['liblinear']
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9]
        },
        "svm": {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ['linear', 'rbf'],
            "model__gamma": ['scale', 'auto']
        }
    }
    
    if XGBClassifier is not None:
        param_distributions["xgboost"] = {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__reg_alpha": [0.1, 1, 10],
            "model__reg_lambda": [0.1, 1, 10]
        }
    
    os.makedirs("data/06_models/classification", exist_ok=True)
    results = []
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    
    for name, estimator in estimators.items():
        print(f"\n🔧 Entrenando {name} ...")
        
        try:
            # Pipeline robusto
            pipe = ImbPipeline([
                ("pre", preprocessor),
                ("smote", sampling_method),
                ("model", estimator)
            ])
            
            # Optimizar por AUC
            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_distributions[name],
                n_iter=15,
                cv=5,  # CV más estable
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
            
            rs.fit(X_train, y_train)
            
            best = rs.best_estimator_
            y_pred = best.predict(X_test)
            y_pred_proba = best.predict_proba(X_test)[:, 1]
            
            # Encontrar mejor threshold para balancear Precision/Recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f1_scores[:-1])  # Excluir el último elemento
            best_threshold = thresholds[best_threshold_idx]
            
            # Usar threshold optimizado
            y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
            
            # Métricas con threshold optimizado
            acc = accuracy_score(y_test, y_pred_optimized)
            f1 = f1_score(y_test, y_pred_optimized)
            prec = precision_score(y_test, y_pred_optimized, zero_division=0)
            rec = recall_score(y_test, y_pred_optimized)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            
            results.append({
                "Model": name,
                "Best_Params": str(rs.best_params_),
                "Accuracy": float(acc),
                "F1_Score": float(f1),
                "Precision": float(prec),
                "Recall": float(rec),
                "AUC_ROC": float(auc),
                "Best_Threshold": float(best_threshold),
                "CV_AUC_Mean": float(cv_mean),
                "CV_AUC_Std": float(cv_std)
            })
            
            joblib.dump(best, f"data/06_models/classification/{name}.pkl")
            
            print(f"[{name}] AUC={auc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
            print(f"[{name}] Best threshold: {best_threshold:.4f}")
            print(f"[{name}] CV_AUC={cv_mean:.3f}±{cv_std:.3f}")
            
        except Exception as e:
            print(f"❌ Error entrenando {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(results)
    
    # Mostrar resultados ordenados
    print("\n" + "="*60)
    print("RESULTADOS FINALES CLASIFICACIÓN (ordenados por AUC):")
    print("="*60)
    print(results_df.sort_values("AUC_ROC", ascending=False).to_string(index=False))
    
    return results_df