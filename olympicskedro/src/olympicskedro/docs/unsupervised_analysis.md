# Análisis No Supervisado - Olympics Dataset

## 🎯 Objetivos
1. Segmentar atletas en grupos naturales usando clustering
2. Reducir dimensionalidad para visualización y análisis
3. Detectar patrones atípicos y anomalías
4. Integrar insights no supervisados con modelos supervisados

## 🔧 Técnicas Implementadas

### Clustering
- **K-Means**: 5 clusters identificados
- **DBSCAN**: 4 clusters + detección de outliers
- **Clustering Jerárquico**: 5 grupos jerárquicos

### Reducción Dimensional
- **PCA**: 85%+ varianza explicada
- **t-SNE**: Visualización no lineal

### Detección de Anomalías
- **Isolation Forest**: Patrones atípicos en participación

## 📊 Resultados

### Calidad de Clustering
| Algoritmo | N Clusters | Silhouette Score |
|-----------|------------|------------------|
| K-Means | 5 | 0.45 |
| DBSCAN | 4 + noise | 0.38 |
| Jerárquico | 5 | 0.42 |

## 💡 Insights de Negocio
- **Cluster 1**: Jóvenes promesas (alta tasa de medallas)
- **Cluster 2**: Veteranos exitosos (experiencia consistente)
- **Cluster 3**: Participantes regulares (rendimiento medio)