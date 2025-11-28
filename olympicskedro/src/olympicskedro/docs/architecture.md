# Arquitectura del Sistema - Olympics ML Pipeline

## 📋 Descripción General
Sistema integral de Machine Learning para análisis de datos olímpicos que combina técnicas supervisadas y no supervisadas.

## 🏗️ Arquitectura del Proyecto

### Estructura de Directorios
proyecto-ml-final/
├── conf/
│ ├── base/
│ │ ├── catalog.yml # Definición de datasets
│ │ └── parameters.yml # Parámetros configurables
├── data/
│ ├── 01_raw/ # Datos originales
│ ├── 02_intermediate/ # Datos procesados
│ ├── 06_models/ # Modelos entrenados
│ ├── 07_model_output/ # Resultados de modelos
│ └── 08_reporting/ # Reportes y métricas
├── src/olympicskedro/pipelines/
│ ├── data_engineering/ # Ingeniería de datos
│ ├── classification/ # Modelos de clasificación
│ ├── regression/ # Modelos de regresión
│ ├── unsupervised_learning/ # Análisis no supervisado
│ ├── integration/ # Integración supervisado + no supervisado
│ ├── pattern_analysis/ # Análisis de patrones
│ └── reporting/ # Generación de reportes
└── notebooks/ # Análisis exploratorios


### 🔄 Flujo de Datos
1. **Ingesta**: Datos originales de Olympics (summer.csv, winter.csv, dictionary.csv)
2. **Preprocesamiento**: Limpieza, transformación y feature engineering
3. **Modelado Supervisado**: Clasificación y regresión
4. **Modelado No Supervisado**: Clustering, reducción dimensional, detección de anomalías
5. **Integración**: Combinación de técnicas supervisadas y no supervisadas
6. **Reporting**: Generación de métricas y visualizaciones

## 🛠️ Stack Tecnológico
- **Framework**: Kedro
- **Orquestación**: Apache Airflow
- **Versionado**: DVC + Git
- **Contenedores**: Docker + Docker Compose
- **ML**: Scikit-learn, XGBoost, LightGBM