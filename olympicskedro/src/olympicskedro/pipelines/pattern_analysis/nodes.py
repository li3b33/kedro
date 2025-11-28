import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
from pathlib import Path

# Custom JSON encoder para manejar tipos no serializables
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

def analyze_cluster_patterns(summer_data: pd.DataFrame, clustering_results: pd.DataFrame, pca_info: Dict) -> pd.DataFrame:
    """
    Análisis profundo de patrones por cluster con interpretación de negocio
    """
    print("Realizando análisis de patrones por cluster...")
    
    # Verificar columnas disponibles
    print(f"Columnas en summer_data: {summer_data.columns.tolist()}")
    print(f"Columnas en clustering_results: {clustering_results.columns.tolist()}")
    
    # Combinar datos originales con resultados de clustering
    analysis_data = summer_data.copy().reset_index(drop=True)
    clustering_data = clustering_results.reset_index(drop=True)
    
    # Asegurar misma longitud
    min_len = min(len(analysis_data), len(clustering_data))
    analysis_data = analysis_data.head(min_len)
    clustering_data = clustering_data.head(min_len)
    
    combined_data = pd.concat([analysis_data, clustering_data], axis=1)
    
    # Análisis por algoritmo de clustering
    cluster_analyses = {}
    
    # 1. ANÁLISIS K-MEANS
    if 'kmeans_cluster' in combined_data.columns:
        kmeans_analysis = analyze_kmeans_clusters(combined_data)
        cluster_analyses['kmeans'] = kmeans_analysis
    
    # 2. ANÁLISIS DBSCAN  
    if 'dbscan_cluster' in combined_data.columns:
        dbscan_analysis = analyze_dbscan_clusters(combined_data)
        cluster_analyses['dbscan'] = dbscan_analysis
    
    # 3. ANÁLISIS JERÁRQUICO
    if 'hierarchical_cluster' in combined_data.columns:
        hierarchical_analysis = analyze_hierarchical_clusters(combined_data)
        cluster_analyses['hierarchical'] = hierarchical_analysis
    
    # 4. ANÁLISIS COMPARATIVO
    comparative_analysis = perform_comparative_analysis(combined_data, cluster_analyses)
    
    # 5. ETIQUETADO SEMÁNTICO
    semantic_labels = generate_semantic_labels(cluster_analyses)
    
    # Guardar análisis completo
    save_pattern_analysis_report(cluster_analyses, comparative_analysis, semantic_labels, pca_info)
    
    print("✅ Análisis de patrones completado")
    
    return combined_data

def analyze_kmeans_clusters(data: pd.DataFrame) -> Dict:
    """Análisis específico para clusters K-Means"""
    analysis = {}
    
    cluster_col = 'kmeans_cluster'
    if cluster_col not in data.columns:
        return analysis
    
    # OBTENER COLUMNAS NUMÉRICAS DISPONIBLES
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    available_numeric = [col for col in numeric_columns if col not in [cluster_col, 'dbscan_cluster', 'hierarchical_cluster', 'anomaly_score', 'is_anomaly']]
    
    # Seleccionar hasta 3 columnas numéricas para análisis
    analysis_columns = available_numeric[:3] if available_numeric else []
    
    # Estadísticas básicas por cluster (solo con columnas disponibles)
    cluster_stats = {}
    
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        cluster_stats[f'cluster_{cluster}'] = {
            'size': int(len(cluster_data))
        }
        
        # Agregar estadísticas de columnas numéricas
        for col in analysis_columns:
            if col in cluster_data.columns:
                cluster_stats[f'cluster_{cluster}'][f'{col}_mean'] = float(cluster_data[col].mean())
                cluster_stats[f'cluster_{cluster}'][f'{col}_std'] = float(cluster_data[col].std())
                cluster_stats[f'cluster_{cluster}'][f'{col}_count'] = int(cluster_data[col].count())
        
        # Agregar HasMedal si existe
        if 'HasMedal' in cluster_data.columns:
            cluster_stats[f'cluster_{cluster}']['hasmedal_mean'] = float(cluster_data['HasMedal'].mean())
        
        # Agregar anomaly_score si existe
        if 'anomaly_score' in cluster_data.columns:
            cluster_stats[f'cluster_{cluster}']['anomaly_score_mean'] = float(cluster_data['anomaly_score'].mean())
    
    # Porcentaje de medallas por cluster
    if 'HasMedal' in data.columns:
        medal_analysis = data.groupby(cluster_col)['HasMedal'].mean().round(4)
        medal_analysis = {f'cluster_{k}': float(v) for k, v in medal_analysis.to_dict().items()}
    else:
        medal_analysis = {f'cluster_{cluster}': 0.0 for cluster in data[cluster_col].unique()}
    
    # Análisis de características de cluster
    cluster_characteristics = {}
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        characteristics = {
            'size': int(len(cluster_data)),
            'has_medal_rate': float(cluster_data['HasMedal'].mean()) if 'HasMedal' in cluster_data.columns else 0.0
        }
        
        # Agregar estadísticas de columnas numéricas
        for col in analysis_columns:
            if col in cluster_data.columns:
                characteristics[f'{col}_mean'] = float(cluster_data[col].mean())
                characteristics[f'{col}_std'] = float(cluster_data[col].std())
        
        cluster_characteristics[f'cluster_{cluster}'] = characteristics
    
    analysis = {
        'cluster_stats': cluster_stats,
        'medal_rates': medal_analysis,
        'cluster_characteristics': cluster_characteristics,
        'cluster_sizes': {f'cluster_{k}': int(v) for k, v in data[cluster_col].value_counts().to_dict().items()},
        'interpretation': interpret_kmeans_clusters(data, cluster_col, analysis_columns)
    }
    
    return analysis

def analyze_dbscan_clusters(data: pd.DataFrame) -> Dict:
    """Análisis específico para clusters DBSCAN"""
    analysis = {}
    
    cluster_col = 'dbscan_cluster'
    if cluster_col not in data.columns:
        return analysis
    
    # DBSCAN tiene clusters (-1 es ruido)
    noise_cluster = data[data[cluster_col] == -1]
    real_clusters = data[data[cluster_col] != -1]
    
    # Características del ruido
    noise_characteristics = {}
    if len(noise_cluster) > 0:
        noise_characteristics = {
            'size': int(len(noise_cluster)),
            'has_medal_rate': float(noise_cluster['HasMedal'].mean()) if 'HasMedal' in noise_cluster.columns else 0.0,
            'anomaly_score_mean': float(noise_cluster['anomaly_score'].mean()) if 'anomaly_score' in noise_cluster.columns else 0.0
        }
    
    # Convertir cluster sizes a formato serializable
    cluster_sizes = {}
    for cluster, size in data[cluster_col].value_counts().to_dict().items():
        cluster_sizes[f'cluster_{cluster}'] = int(size)
    
    analysis = {
        'noise_points': int(len(noise_cluster)),
        'noise_ratio': float(len(noise_cluster) / len(data)),
        'real_clusters_count': int(real_clusters[cluster_col].nunique()),
        'cluster_sizes': cluster_sizes,
        'noise_characteristics': noise_characteristics,
        'interpretation': interpret_dbscan_clusters(data, cluster_col)
    }
    
    return analysis

def analyze_hierarchical_clusters(data: pd.DataFrame) -> Dict:
    """Análisis específico para clusters jerárquicos"""
    analysis = {}
    
    cluster_col = 'hierarchical_cluster'
    if cluster_col not in data.columns:
        return analysis
    
    # Obtener columnas numéricas disponibles
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    available_numeric = [col for col in numeric_columns if col not in [cluster_col, 'dbscan_cluster', 'kmeans_cluster', 'anomaly_score', 'is_anomaly']]
    analysis_columns = available_numeric[:2] if available_numeric else []
    
    # Estadísticas básicas por cluster
    cluster_stats = {}
    
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        cluster_stats[f'cluster_{cluster}'] = {
            'size': int(len(cluster_data))
        }
        
        for col in analysis_columns:
            if col in cluster_data.columns:
                cluster_stats[f'cluster_{cluster}'][f'{col}_mean'] = float(cluster_data[col].mean())
                cluster_stats[f'cluster_{cluster}'][f'{col}_std'] = float(cluster_data[col].std())
        
        if 'HasMedal' in cluster_data.columns:
            cluster_stats[f'cluster_{cluster}']['hasmedal_mean'] = float(cluster_data['HasMedal'].mean())
    
    # Convertir cluster sizes a formato serializable
    cluster_sizes = {}
    for cluster, size in data[cluster_col].value_counts().to_dict().items():
        cluster_sizes[f'cluster_{cluster}'] = int(size)
    
    analysis = {
        'cluster_stats': cluster_stats,
        'cluster_sizes': cluster_sizes,
        'interpretation': interpret_hierarchical_clusters(data, cluster_col)
    }
    
    return analysis

def interpret_kmeans_clusters(data: pd.DataFrame, cluster_col: str, numeric_columns: List[str]) -> Dict:
    """Interpretación de negocio para clusters K-Means"""
    interpretations = {}
    
    for cluster in sorted(data[cluster_col].unique()):
        cluster_data = data[data[cluster_col] == cluster]
        
        # Características del cluster
        medal_rate = float(cluster_data['HasMedal'].mean()) if 'HasMedal' in cluster_data.columns else 0.0
        
        # Usar la primera columna numérica disponible para interpretación
        main_feature = None
        main_feature_value = None
        
        if numeric_columns and len(numeric_columns) > 0:
            main_feature = str(numeric_columns[0])
            main_feature_value = float(cluster_data[main_feature].mean()) if main_feature in cluster_data.columns else None
        
        # Interpretación semántica basada en características disponibles
        if medal_rate > 0.3:
            if main_feature_value and main_feature_value > data[main_feature].median() if main_feature in data.columns else False:
                label = "Alto Rendimiento"
            else:
                label = "Éxito Consistente"
        elif medal_rate < 0.1:
            label = "Participación Básica"
        else:
            label = "Rendimiento Regular"
        
        interpretations[f'cluster_{cluster}'] = {
            'semantic_label': str(label),
            'medal_rate': float(round(medal_rate, 4)),
            'size': int(len(cluster_data)),
            'main_characteristic': main_feature,
            'main_value': float(round(main_feature_value, 2)) if main_feature_value else None
        }
    
    return interpretations

def interpret_dbscan_clusters(data: pd.DataFrame, cluster_col: str) -> Dict:
    """Interpretación de negocio para clusters DBSCAN"""
    interpretations = {}
    
    # Cluster de ruido
    noise_data = data[data[cluster_col] == -1]
    if len(noise_data) > 0:
        medal_rate = float(noise_data['HasMedal'].mean()) if 'HasMedal' in noise_data.columns else 0.0
        
        if medal_rate > 0.4:
            noise_label = "Casos de Éxito Extremo"
        elif medal_rate < 0.05:
            noise_label = "Casos de Bajo Rendimiento"
        else:
            noise_label = "Patrones Inusuales"
        
        interpretations['noise_cluster'] = {
            'semantic_label': str(noise_label),
            'size': int(len(noise_data)),
            'medal_rate': float(round(medal_rate, 4)),
            'characteristics': 'Patrones atípicos o datos extremos'
        }
    
    # Clusters reales
    real_clusters = data[data[cluster_col] != -1][cluster_col].unique()
    for cluster in real_clusters:
        cluster_data = data[data[cluster_col] == cluster]
        medal_rate = float(cluster_data['HasMedal'].mean()) if 'HasMedal' in cluster_data.columns else 0.0
        
        if medal_rate > 0.3:
            cluster_label = f'Grupo Cohesionado de Alto Rendimiento {cluster}'
        else:
            cluster_label = f'Grupo Cohesionado Regular {cluster}'
        
        interpretations[f'cluster_{cluster}'] = {
            'semantic_label': str(cluster_label),
            'size': int(len(cluster_data)),
            'medal_rate': float(round(medal_rate, 4)),
            'density_characteristics': 'Alta densidad de puntos similares'
        }
    
    return interpretations

def interpret_hierarchical_clusters(data: pd.DataFrame, cluster_col: str) -> Dict:
    """Interpretación de negocio para clusters jerárquicos"""
    interpretations = {}
    
    for cluster in sorted(data[cluster_col].unique()):
        cluster_data = data[data[cluster_col] == cluster]
        medal_rate = float(cluster_data['HasMedal'].mean()) if 'HasMedal' in cluster_data.columns else 0.0
        
        if medal_rate > 0.35:
            label = f'Grupo Jerárquico de Élite {cluster}'
        elif medal_rate > 0.15:
            label = f'Grupo Jerárquico Competitivo {cluster}'
        else:
            label = f'Grupo Jerárquico Base {cluster}'
        
        interpretations[f'cluster_{cluster}'] = {
            'semantic_label': str(label),
            'size': int(len(cluster_data)),
            'medal_rate': float(round(medal_rate, 4)),
            'hierarchical_level': 'Agrupamiento estructural'
        }
    
    return interpretations

def perform_comparative_analysis(data: pd.DataFrame, cluster_analyses: Dict) -> Dict:
    """Análisis comparativo entre diferentes algoritmos"""
    comparative = {}
    
    # Comparar tamaños de cluster entre algoritmos
    cluster_sizes_comparison = {}
    for algo, analysis in cluster_analyses.items():
        if 'cluster_sizes' in analysis:
            cluster_sizes_comparison[algo] = analysis['cluster_sizes']
    
    # Análisis de consistencia entre algoritmos
    consistency_analysis = analyze_algorithm_consistency(data)
    
    comparative = {
        'cluster_sizes_comparison': cluster_sizes_comparison,
        'algorithm_consistency': consistency_analysis,
        'recommended_algorithm': str(recommend_best_algorithm(cluster_analyses))
    }
    
    return comparative

def analyze_algorithm_consistency(data: pd.DataFrame) -> Dict:
    """Analiza consistencia entre diferentes algoritmos de clustering"""
    consistency = {}
    
    # Verificar correlación entre asignaciones de cluster
    cluster_cols = ['kmeans_cluster', 'dbscan_cluster', 'hierarchical_cluster']
    available_cols = [col for col in cluster_cols if col in data.columns]
    
    if len(available_cols) >= 2:
        try:
            from scipy.stats import chi2_contingency
            
            for i, col1 in enumerate(available_cols):
                for col2 in available_cols[i+1:]:
                    contingency_table = pd.crosstab(data[col1], data[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    consistency[f'{col1}_vs_{col2}'] = {
                        'chi2_statistic': float(round(chi2, 4)),
                        'p_value': float(round(p_value, 6)),
                        'significant': bool(p_value < 0.05)
                    }
        except ImportError:
            print("⚠️  scipy no disponible para análisis de consistencia")
    
    return consistency

def recommend_best_algorithm(cluster_analyses: Dict) -> str:
    """Recomienda el mejor algoritmo basado en el análisis"""
    if not cluster_analyses:
        return "No hay algoritmos para comparar"
    
    # Evaluar basado en interpretabilidad y características
    scores = {}
    
    for algo, analysis in cluster_analyses.items():
        score = 0
        
        # Puntos por interpretaciones semánticas
        if 'interpretation' in analysis:
            score += len(analysis['interpretation']) * 2
        
        # Puntos por análisis detallado
        if 'cluster_stats' in analysis:
            score += 3
        
        # Puntos por identificación de ruido (DBSCAN)
        if algo == 'dbscan' and 'noise_points' in analysis:
            score += 2
        
        scores[algo] = score
    
    best_algo = max(scores, key=scores.get) if scores else "kmeans"
    return str(best_algo)

def generate_semantic_labels(cluster_analyses: Dict) -> Dict:
    """Genera etiquetas semánticas para clusters"""
    semantic_labels = {}
    
    for algo, analysis in cluster_analyses.items():
        if 'interpretation' in analysis:
            algo_labels = {}
            for cluster_key, cluster_info in analysis['interpretation'].items():
                if 'semantic_label' in cluster_info:
                    algo_labels[str(cluster_key)] = str(cluster_info['semantic_label'])
            
            semantic_labels[str(algo)] = algo_labels
    
    return semantic_labels

def save_pattern_analysis_report(cluster_analyses: Dict, comparative_analysis: Dict, 
                               semantic_labels: Dict, pca_info: Dict) -> None:
    """Guarda reporte completo del análisis de patrones"""
    
    # Convertir pca_info a formato serializable si es necesario
    serializable_pca_info = {}
    if isinstance(pca_info, dict):
        serializable_pca_info = pca_info
    elif hasattr(pca_info, 'to_dict'):
        serializable_pca_info = pca_info.to_dict()
    else:
        # Si es un DataFrame, tomar solo la información relevante
        serializable_pca_info = {
            'shape': str(getattr(pca_info, 'shape', 'unknown')),
            'columns': [str(col) for col in getattr(pca_info, 'columns', [])] if hasattr(pca_info, 'columns') else []
        }
    
    report = {
        'timestamp': str(pd.Timestamp.now().isoformat()),
        'cluster_analyses': cluster_analyses,
        'comparative_analysis': comparative_analysis,
        'semantic_labels': semantic_labels,
        'pca_info': serializable_pca_info
    }
    
    # Guardar como JSON
    output_path = Path("data/08_reporting/pattern_analysis_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
    
    print(f"✅ Reporte de análisis de patrones guardado en: {output_path}")
    
    # También guardar resumen ejecutivo
    save_executive_summary(report)

def save_executive_summary(report: Dict) -> None:
    """Guarda un resumen ejecutivo del análisis"""
    summary = {
        'total_clusters_analyzed': int(sum(
            len(analysis.get('interpretation', {})) 
            for analysis in report['cluster_analyses'].values()
        )),
        'best_algorithm': str(report['comparative_analysis'].get('recommended_algorithm', 'N/A')),
        'key_insights': [str(insight) for insight in extract_key_insights(report)],
        'business_recommendations': [str(rec) for rec in generate_business_recommendations(report)]
    }
    
    summary_path = Path("data/08_reporting/pattern_analysis_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
    
    print(f"✅ Resumen ejecutivo guardado en: {summary_path}")

def extract_key_insights(report: Dict) -> List[str]:
    """Extrae insights clave del análisis"""
    insights = []
    
    # Insight 1: Mejor algoritmo
    best_algo = report['comparative_analysis'].get('recommended_algorithm', 'N/A')
    insights.append(f"Algoritmo recomendado: {best_algo.upper()} basado en interpretabilidad")
    
    # Insight 2: Patrones de medallas
    for algo, analysis in report['cluster_analyses'].items():
        if 'interpretation' in analysis:
            for cluster_key, cluster_info in analysis['interpretation'].items():
                if cluster_info.get('medal_rate', 0) > 0.3:
                    insights.append(f"Cluster {algo} ({cluster_key}) tiene alta tasa de medallas: {cluster_info['medal_rate']}")
    
    # Insight 3: Grupos identificados
    total_clusters = sum(
        len(analysis.get('interpretation', {})) 
        for analysis in report['cluster_analyses'].values()
    )
    insights.append(f"Se identificaron {total_clusters} grupos distintos de atletas")
    
    return insights[:5]  # Limitar a 5 insights principales

def generate_business_recommendations(report: Dict) -> List[str]:
    """Genera recomendaciones de negocio basadas en el análisis"""
    recommendations = []
    
    recommendations.append("ENFOQUE EN TALENTO: Identificar clusters con alta tasa de medallas para programas de desarrollo")
    recommendations.append("DETECCIÓN TEMPRANA: Usar clustering para identificar patrones de atletas prometedores")
    recommendations.append("PERSONALIZACIÓN: Desarrollar programas de entrenamiento específicos por perfil de cluster")
    recommendations.append("GESTIÓN DE RENDIMIENTO: Monitorear clusters de bajo rendimiento para intervenciones tempranas")
    
    return recommendations