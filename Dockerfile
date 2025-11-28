FROM apache/airflow:2.8.1-python3.11

# Cambiar a root temporalmente para instalar dependencias del sistema
USER root
RUN apt-get update && apt-get install -y build-essential git && apt-get clean

# Directorio base dentro del contenedor
WORKDIR /opt/airflow/project

# Copiamos el proyecto Kedro completo
COPY olympicskedro/ ./olympicskedro/
COPY olympicskedro/conf/ ./olympicskedro/conf/
COPY olympicskedro/pyproject.toml ./olympicskedro/pyproject.toml

# Copiamos otros archivos del proyecto que podrían ser necesarios
COPY requirements.txt .
COPY README.md .

# Cambiamos a usuario airflow antes de instalar dependencias
USER airflow

# Instalamos las dependencias Python del proyecto
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir umap-learn pyod mixtend hdbscan shap plotly

# Instalar el proyecto Kedro como paquete
RUN pip install -e ./olympicskedro

WORKDIR /opt/airflow