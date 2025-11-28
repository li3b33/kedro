import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def generate_clustering_summary(clustering_results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Genera resumen ejecutivo del clustering"""
    
    summary_data = []
    
    for _, row in metrics.iterrows():
        algorithm = row['algorithm']
        n_clusters = row['n_clusters']
        silhouette = row.get('silhouette_score', 0)
        
        # Estadísticas por algoritmo
        cluster_col = f"{algorithm.lower()}_cluster"
        if cluster_col in clustering_results.columns:
            cluster_dist = clustering_results[cluster_col].value_counts().to_dict()
            
            summary_data.append({
                'Algorithm': algorithm,
                'N_Clusters': n_clusters,
                'Silhouette_Score': round(silhouette, 4),
                'Davies_Bouldin': round(row.get('davies_bouldin_score', 0), 4),
                'Calinski_Harabasz': round(row.get('calinski_harabasz_score', 0), 4),
                'Largest_Cluster_Size': max(cluster_dist.values()) if cluster_dist else 0,
                'Anomalies_Detected': clustering_results['is_anomaly'].sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_visualization_data(clustering_results: pd.DataFrame, pca_info: Dict[str, Any]) -> pd.DataFrame:
    """Prepara datos para visualizaciones"""
    
    viz_data = clustering_results.copy()
    
    # Agregar información de PCA
    if 'pca_info' in pca_info:
        viz_data['pca_variance_explained'] = pca_info['pca_info']['cumulative_variance'][-1]
    
    return viz_data

def generate_unsupervised_report(clustering_summary: pd.DataFrame, visualization_data: pd.DataFrame) -> pd.DataFrame:
    """Genera reporte final integrado"""
    
    report_data = {
        'Total_Samples': len(visualization_data),
        'Total_Anomalies': visualization_data['is_anomaly'].sum(),
        'Best_Clustering_Algorithm': clustering_summary.loc[clustering_summary['Silhouette_Score'].idxmax(), 'Algorithm'],
        'Best_Silhouette_Score': clustering_summary['Silhouette_Score'].max(),
        'PCA_Variance_Explained': visualization_data['pca_variance_explained'].iloc[0] if 'pca_variance_explained' in visualization_data.columns else 0
    }
    
    report_df = pd.DataFrame([report_data])
    return report_df