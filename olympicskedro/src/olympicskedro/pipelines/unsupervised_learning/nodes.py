import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import joblib
import json
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

def prepare_clustering_data(summer_data: pd.DataFrame, dictionary_data: pd.DataFrame) -> pd.DataFrame:
    print("Preparando datos para clustering...")
    
    summer_df = summer_data.copy()
    dict_df = dictionary_data.copy()
    
    print(f"Columnas en summer_data: {summer_df.columns.tolist()}")
    print(f"Columnas en dictionary_data: {dict_df.columns.tolist()}")
    
    available_columns_summer = summer_df.columns.tolist()
    available_columns_dict = dict_df.columns.tolist()
    
    clustering_features = []
    
    if 'Year' in available_columns_summer:
        clustering_features.append('Year')
    
    if 'Age' in available_columns_summer:
        clustering_features.append('Age')
    
    if 'HasMedal' in available_columns_summer:
        clustering_features.append('HasMedal')
    
    if 'Sport' in available_columns_summer:
        sport_le = LabelEncoder()
        summer_df['sport_encoded'] = sport_le.fit_transform(summer_df['Sport'].astype(str))
        clustering_features.append('sport_encoded')
    
    if 'Country' in available_columns_summer:
        country_le = LabelEncoder()
        summer_df['country_encoded'] = country_le.fit_transform(summer_df['Country'].astype(str))
        clustering_features.append('country_encoded')
    
    if 'Gender' in available_columns_summer:
        gender_le = LabelEncoder()
        summer_df['gender_encoded'] = gender_le.fit_transform(summer_df['Gender'].astype(str))
        clustering_features.append('gender_encoded')
    
    if 'GDP per Capita' in available_columns_dict:
        if len(dict_df) == len(summer_df):
            summer_df['gdp_per_capita'] = dict_df['GDP per Capita'].values
            clustering_features.append('gdp_per_capita')
        else:
            print("No se puede mapear GDP per Capita - diferentes tamaños de dataset")
    
    if 'Population' in available_columns_dict:
        if len(dict_df) == len(summer_df):
            summer_df['population'] = dict_df['Population'].values
            clustering_features.append('population')
        else:
            print("No se puede mapear Population - diferentes tamaños de dataset")
    
    if 'Athlete' in available_columns_summer:
        athlete_stats = summer_df.groupby('Athlete').agg({
            'Year': ['min', 'max', 'nunique'],
            'Sport': 'nunique',
            'Country': 'first',
            'HasMedal': 'sum'
        }).reset_index()
        
        athlete_stats.columns = ['Athlete', 'year_first', 'year_last', 'year_count', 'sport_count', 'country', 'medal_count']
        athlete_stats['career_length'] = athlete_stats['year_last'] - athlete_stats['year_first']
        athlete_stats['medal_rate'] = athlete_stats['medal_count'] / athlete_stats['year_count']
        
        summer_df = summer_df.merge(athlete_stats[['Athlete', 'career_length', 'medal_rate', 'sport_count']], 
                                   on='Athlete', how='left')
        clustering_features.extend(['career_length', 'medal_rate', 'sport_count'])
    
    useful_numeric_cols = [col for col in clustering_features if col in summer_df.columns]
    
    if not useful_numeric_cols:
        all_numeric_cols = summer_df.select_dtypes(include=[np.number]).columns.tolist()
        useful_numeric_cols = all_numeric_cols[:5]
    
    print(f"Columnas numéricas útiles encontradas: {useful_numeric_cols}")
    
    if len(useful_numeric_cols) < 2:
        for col in ['Year', 'HasMedal']:
            if col in summer_df.columns and col not in useful_numeric_cols:
                useful_numeric_cols.append(col)
        
        if len(useful_numeric_cols) < 2:
            summer_df['dummy_feature_1'] = np.random.randn(len(summer_df))
            summer_df['dummy_feature_2'] = np.random.randn(len(summer_df))
            useful_numeric_cols.extend(['dummy_feature_1', 'dummy_feature_2'])
    
    clustering_df = summer_df[useful_numeric_cols].copy()
    
    null_threshold = 0.5
    null_ratio = clustering_df.isnull().sum() / len(clustering_df)
    valid_columns = null_ratio[null_ratio <= null_threshold].index.tolist()
    clustering_df = clustering_df[valid_columns]
    
    clustering_df = clustering_df.fillna(clustering_df.median())
    
    variance_threshold = 0.001
    numeric_variance = clustering_df.var()
    high_variance_columns = numeric_variance[numeric_variance > variance_threshold].index.tolist()
    clustering_df = clustering_df[high_variance_columns]
    
    if len(clustering_df.columns) < 2:
        clustering_df['feature_1'] = np.random.randn(len(clustering_df))
        clustering_df['feature_2'] = np.random.randn(len(clustering_df))
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df)
    clustering_df_scaled = pd.DataFrame(scaled_features, columns=clustering_df.columns)
    
    print(f"Datos finales para clustering: {clustering_df_scaled.shape}")
    print(f"Características: {clustering_df_scaled.columns.tolist()}")
    
    os.makedirs("data/06_models/unsupervised", exist_ok=True)
    joblib.dump(scaler, "data/06_models/unsupervised/scaler.pkl")
    
    return clustering_df_scaled

def perform_kmeans_clustering(features: pd.DataFrame):
    print("Ejecutando K-Means clustering...")
    
    max_k = min(10, len(features) // 10)
    if max_k < 2:
        max_k = 2
    
    k_range = range(2, max_k + 1)
    wcss = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    
    if len(wcss) > 2:
        differences = np.diff(wcss)
        second_differences = np.diff(differences)
        if len(second_differences) > 0:
            optimal_k = k_range[np.argmin(second_differences) + 2]
        else:
            optimal_k = 3
    else:
        optimal_k = 3
    
    optimal_k = min(optimal_k, max_k)
    print(f"K óptimo sugerido: {optimal_k}")
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(features)
    
    if len(np.unique(cluster_labels)) > 1:
        sil_score = silhouette_score(features, cluster_labels)
        db_score = davies_bouldin_score(features, cluster_labels)
        ch_score = calinski_harabasz_score(features, cluster_labels)
    else:
        sil_score = -1
        db_score = float('inf')
        ch_score = 0
    
    metrics = {
        'algorithm': 'KMeans',
        'n_clusters': optimal_k,
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'inertia': kmeans_final.inertia_
    }
    
    return kmeans_final, cluster_labels, metrics, wcss

def perform_dbscan_clustering(features: pd.DataFrame):
    print("Ejecutando DBSCAN clustering...")
    
    best_score = -1
    best_eps = 0.5
    best_min_samples = 5
    best_labels = None
    
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [5, 10, 15]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters < len(features) // 2:
                try:
                    sil_score = silhouette_score(features, labels)
                    if sil_score > best_score:
                        best_score = sil_score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels
                except:
                    continue
    
    if best_labels is None:
        dbscan_final = DBSCAN(eps=0.5, min_samples=5)
        final_labels = dbscan_final.fit_predict(features)
        best_score = -1
    else:
        dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        final_labels = best_labels
    
    n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    n_noise = list(final_labels).count(-1)
    
    metrics = {
        'algorithm': 'DBSCAN',
        'n_clusters': n_clusters,
        'eps': best_eps,
        'min_samples': best_min_samples,
        'silhouette_score': best_score,
        'n_noise': n_noise
    }
    
    return dbscan_final, final_labels, metrics

def perform_hierarchical_clustering(features: pd.DataFrame):
    print("Ejecutando clustering jerárquico...")
    
    if len(features) > 1000:
        sample_indices = np.random.choice(len(features), 1000, replace=False)
        features_sample = features.iloc[sample_indices]
    else:
        features_sample = features
    
    linkage_matrix = linkage(features_sample, method='ward')
    
    if len(linkage_matrix) >= 5:
        last_rev = linkage_matrix[-5:, 2]
        last_rev_diff = np.diff(last_rev)
        if len(last_rev_diff) > 0:
            optimal_k = np.argmax(last_rev_diff) + 2
        else:
            optimal_k = 3
    else:
        optimal_k = 3
    
    optimal_k = min(optimal_k, 10)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    if len(np.unique(cluster_labels)) > 1:
        sil_score = silhouette_score(features, cluster_labels)
        db_score = davies_bouldin_score(features, cluster_labels)
        ch_score = calinski_harabasz_score(features, cluster_labels)
    else:
        sil_score = -1
        db_score = float('inf')
        ch_score = 0
    
    metrics = {
        'algorithm': 'Hierarchical',
        'n_clusters': optimal_k,
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score
    }
    
    return kmeans, cluster_labels, metrics

def perform_pca_analysis(features: pd.DataFrame, n_components: int = 2):
    print("Ejecutando PCA...")
    
    n_components = min(n_components, features.shape[1])
    
    if n_components < 1:
        n_components = 1
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(features)
    
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=pca_columns)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    pca_info = {
        'explained_variance_ratio': [float(x) for x in explained_variance.tolist()],
        'cumulative_variance': [float(x) for x in cumulative_variance.tolist()],
        'components': [[float(y) for y in x] for x in pca.components_.tolist()],
        'features': features.columns.tolist(),
        'n_components': int(n_components)
    }
    
    return pca, pca_df, pca_info

def perform_tsne_analysis(features: pd.DataFrame, n_components: int = 2):
    print("Ejecutando t-SNE...")
    
    if features.shape[1] <= 1:
        tsne_result = np.random.randn(len(features), n_components)
    else:
        if features.shape[1] > 30:
            pca = PCA(n_components=30, random_state=42)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features.values
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(features)-1))
        tsne_result = tsne.fit_transform(features_reduced)
    
    tsne_columns = [f'TSNE_{i+1}' for i in range(n_components)]
    tsne_df = pd.DataFrame(tsne_result, columns=tsne_columns)
    
    return None, tsne_df

def perform_anomaly_detection(features: pd.DataFrame):
    print("Ejecutando detección de anomalías...")
    
    contamination = min(0.1, 0.05 * len(features))
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features)
    anomaly_scores = iso_forest.decision_function(features)
    
    anomaly_binary = (anomaly_labels == -1).astype(int)
    
    anomaly_df = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'is_anomaly': anomaly_binary
    })
    
    n_anomalies = anomaly_binary.sum()
    print(f"Anomalías detectadas: {n_anomalies} ({n_anomalies/len(features)*100:.2f}%)")
    
    return iso_forest, anomaly_df

def run_complete_unsupervised_analysis(summer_data: pd.DataFrame, dictionary_data: pd.DataFrame):
    print("=== INICIANDO ANÁLISIS NO SUPERVISADO COMPLETO ===")
    
    try:
        features = prepare_clustering_data(summer_data, dictionary_data)
        
        if len(features) < 10:
            raise ValueError("Datos insuficientes para clustering")
        
        kmeans_model, kmeans_labels, kmeans_metrics, wcss = perform_kmeans_clustering(features)
        dbscan_model, dbscan_labels, dbscan_metrics = perform_dbscan_clustering(features)
        hierarchical_model, hierarchical_labels, hierarchical_metrics = perform_hierarchical_clustering(features)
        
        pca_model, pca_results, pca_info = perform_pca_analysis(features)
        tsne_model, tsne_results = perform_tsne_analysis(features)
        
        anomaly_model, anomaly_results = perform_anomaly_detection(features)
        
        clustering_results = pd.DataFrame({
            'kmeans_cluster': kmeans_labels,
            'dbscan_cluster': dbscan_labels,
            'hierarchical_cluster': hierarchical_labels,
            'anomaly_score': anomaly_results['anomaly_score'],
            'is_anomaly': anomaly_results['is_anomaly']
        })
        
        clustering_results = pd.concat([clustering_results, pca_results, tsne_results], axis=1)
        
        all_metrics = [kmeans_metrics, dbscan_metrics, hierarchical_metrics]
        metrics_df = pd.DataFrame(all_metrics)
        
        save_unsupervised_metrics(all_metrics)
        
        os.makedirs("data/06_models/unsupervised", exist_ok=True)
        joblib.dump(kmeans_model, "data/06_models/unsupervised/kmeans_model.pkl")
        joblib.dump(dbscan_model, "data/06_models/unsupervised/dbscan_model.pkl")
        joblib.dump(hierarchical_model, "data/06_models/unsupervised/hierarchical_model.pkl")
        joblib.dump(pca_model, "data/06_models/unsupervised/pca_model.pkl")
        joblib.dump(anomaly_model, "data/06_models/unsupervised/isolation_forest_model.pkl")
        
        print("=== ANÁLISIS NO SUPERVISADO COMPLETADO ===")
        
        return {
            'clustering_results': clustering_results,
            'metrics': metrics_df,
            'pca_info': pca_results,
            'models': {
                'kmeans': kmeans_model,
                'dbscan': dbscan_model,
                'hierarchical': hierarchical_model,
                'pca': pca_model,
                'tsne': tsne_model,
                'anomaly': anomaly_model
            },
            'wcss': wcss
        }
    
    except Exception as e:
        print(f"Error en análisis no supervisado: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def save_unsupervised_metrics(metrics_list: list):
    metrics_dict = {}
    
    for metric in metrics_list:
        algo = metric['algorithm'].lower()
        metrics_dict[f"{algo}_silhouette"] = float(round(metric.get('silhouette_score', 0), 4))
        metrics_dict[f"{algo}_davies_bouldin"] = float(round(metric.get('davies_bouldin_score', 0), 4))
        metrics_dict[f"{algo}_calinski_harabasz"] = float(round(metric.get('calinski_harabasz_score', 0), 4))
        metrics_dict[f"{algo}_n_clusters"] = int(metric.get('n_clusters', 0))
    
    metrics_output = Path("data/08_reporting/unsupervised_metrics_flat.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_output, "w") as f:
        json.dump(metrics_dict, f, indent=4, sort_keys=True)
    
    print(f"Métricas guardadas en: {metrics_output}")
    
    return metrics_dict