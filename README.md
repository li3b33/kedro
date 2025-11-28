# OlympicsKedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)


## Dataset kaggle
[![Dataset](https://img.shields.io/badge/-Dataset_Olympics_Games-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/the-guardian/olympic-games)



## Video presentación Ev2 
[![Video](https://img.shields.io/badge/-Video_Explicativo_OlympicsKedro-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1YgcR6Xv4p7QSginYgnExgHLqoo-wqFcM/view?usp=sharing)


## 🐍 Crear entorno virtual

Antes de comenzar, crea un entorno virtual llamado `venv` y actívalo:

```bash
python -m venv venv
source venv/bin/activate  #En Linux
venv\Scripts\activate  #En Windows
```

Una vez activado, puedes instalar las dependencias como se indica más abajo.

---

## 🧱 Descripción general del proyecto

Este proyecto implementa un pipeline de Machine Learning sobre datos históricos de los Juegos Olímpicos utilizando:

**Kedro** → para estructurar y ejecutar pipelines de datos reproducibles

**DVC** → para versionar datasets y modelos

**Airflow** → para orquestar la ejecución automatizada de pipelines

**Docker** → para desplegar y ejecutar todo el ecosistema en contenedores

## El proyecto incluye pipelines de:

**Data Engineering** (preprocesamiento y limpieza)

**Classification** (5 modelos de clasificación)

**Regression** (modelos de predicción continua)

**Unsupervised Learning** (clustering, reducción dimensional, anomalías)

**Integration** (combinación supervisado + no supervisado)

**Pattern Analysis** (análisis de patrones por cluster)

**Reporting** (generación de métricas y resultados)

---

## 📌 Reglas y pautas

Para sacar el máximo provecho de esta plantilla:

* No elimines ninguna línea del archivo `.gitignore` proporcionado.
* Asegúrate de que tus resultados puedan ser reproducidos siguiendo una convención de ingeniería de datos.
* **No subas datos** a tu repositorio.
* **No subas credenciales** ni configuraciones locales. Guarda todo eso en el directorio `conf/local/`.

---

## 📦 Cómo instalar las dependencias

Declara las dependencias necesarias en el archivo `requirements.txt`.

Para instalarlas, ejecuta:

```
pip install -r requirements.txt
```

---

## ▶️ Cómo ejecutar tu pipeline de Kedro

Puedes ejecutar tu proyecto Kedro con:

```
kedro run
```

## Ejecutar pipelines específicos

```
🔧 Pipelines Individuales:
kedro run --pipeline=data_engineering
kedro run --pipeline=classification
kedro run --pipeline=regression
kedro run --pipeline=unsupervised
kedro run --pipeline=integration
kedro run --pipeline=pattern_analysis
kedro run --pipeline=innovation
kedro run --pipeline=reporting
kedro run --pipeline=reporting_unsupervised

🚀 Pipelines Combinados:
kedro run --pipeline=supervised_learning (clasificación + regresión)
kedro run --pipeline=ml_pipelines (supervisado + no supervisado)
kedro run --pipeline=analysis_pipelines (no supervisado + análisis patrones + reporting)
kedro run --pipeline=advanced_analysis (no supervisado + análisis patrones + innovación)
kedro run --pipeline=complete_analysis (análisis completo con innovación)
kedro run --pipeline=complete_modeling (modelado completo)
kedro run --pipeline=demo_pipeline (demostración rápida)

⚡ Pipelines Rápidos:
kedro run --pipeline=quick_test (solo data engineering + clasificación)
kedro run --pipeline=data_processing (solo procesamiento de datos)
kedro run --pipeline=model_training (solo entrenamiento de modelos)

🎯 Pipeline Completo:
kedro run o kedro run --pipeline=full_pipeline (EJECUTA TODO)

🔧 Pipelines de Desarrollo:
kedro run --pipeline=full_without_reporting (todo excepto reporting)
kedro run --pipeline=full_without_innovation (todo excepto innovación)

```

---

## 💾 Control de versiones con DVC

El proyecto utiliza DVC (Data Version Control) para rastrear datasets y modelos.
Pasos básicos:

```
dvc init
dvc add data/01_raw data/02_intermediate data/07_model_output
git add .
git commit -m "Track data with DVC"
```

Para guardar versiones de los datos:

```
dvc push
```

---

## ☁️ Orquestación con Apache Airflow

El DAG principal se llama olympicskedro_pipeline
y se encuentra en:

```
airflow_dags/olympicskedro_dag.py
```

Levantar Airflow con Docker

Asegúrate de tener Docker corriendo y ejecuta:

```
docker compose up -d
```

Accede a la interfaz web:

👉 http://localhost:8080

Credenciales por defecto:

**Usuario:** admin

**Contraseña:** admin


Ejecuta manualmente el DAG desde la interfaz para correr todo el pipeline Kedro dentro de los contenedores.

---

## 🐳 Docker

[![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com)


El proyecto se ejecuta dentro de un entorno Dockerizado.

Construir las imágenes

```
docker compose build --no-cache
```

Iniciar todos los servicios

```
docker compose up -d
```

Esto levantará:

- PostgreSQL (base de datos de Airflow)

- Airflow Webserver

- Airflow Scheduler

---
## 📊 Resultados y Métricas

### Modelos de Clasificación (Predicción de Medallas)
* Mejor modelo: Logistic Regression (AUC: 0.516)
* Algoritmos probados: Random Forest, XGBoost, Gradient Boosting, LightGBM, SVM
* Métricas: AUC-ROC, F1-Score, Precision, Recall, Accuracy

### Modelos de Regresión (Predicción de GDP)
* Mejor modelo: Ensemble (R²: > 0.7)
* Algoritmos: Random Forest, Gradient Boosting, Ridge, Lasso, XGBoost
* Métricas: RMSE, MAE, R², MAPE

### Análisis No Supervisado
* Clustering: 3 algoritmos implementados (K-Means, DBSCAN, Jerárquico)
* Reducción dimensional: PCA y t-SNE
* Detección de anomalías: Isolation Forest
* Métricas: Silhouette Score, Davies-Bouldin, Calinski-Harabasz

### Integración Supervisado + No Supervisado
* Enfoque: Clusters como features para modelos supervisados
* Resultado: Análisis comparativo de mejora de rendimiento
---
## 🏗️ Arquitectura del Proyecto
```
olympicskedro/
├── conf/                    # Configuración
│   ├── base/               # Configuración base
│   │   ├── catalog.yml     # Definición de datasets
│   │   └── parameters.yml  # Parámetros configurables
├── data/                   # Datos y modelos
│   ├── 01_raw/             # Datos originales
│   ├── 02_intermediate/    # Datos procesados  
│   ├── 06_models/          # Modelos entrenados
│   ├── 07_model_output/    # Resultados de modelos
│   └── 08_reporting/       # Reportes y métricas
├── docs/                   # Documentación técnica
├── notebooks/              # Análisis exploratorios
├── src/olympicskedro/pipelines/  # Pipelines de procesamiento
│   ├── data_engineering/   # Ingeniería de datos
│   ├── classification/     # Modelos de clasificación
│   ├── regression/         # Modelos de regresión
│   ├── unsupervised_learning/     # Análisis no supervisado
│   ├── integration/        # Integración supervisado + no supervisado
│   ├── pattern_analysis/   # Análisis de patrones
│   └── reporting/          # Generación de reportes
├── airflow_dags/           # Orquestación con Airflow
└── docker/                 # Configuración Docker
```

---

## 🔄 Flujo de Datos
1. Ingesta: Datos originales de Olympics (summer.csv, winter.csv, dictionary.csv)
2. Preprocesamiento: Limpieza, transformación y feature engineering
3. Modelado Supervisado: Clasificación y regresión
4. Modelado No Supervisado: Clustering, reducción dimensional, detección de anomalías
5. Integración: Combinación de técnicas supervisadas y no supervisadas
6. Pattern Analysis: Análisis profundo de clusters y patrones
7. Reporting: Generación de métricas y visualizaciones

---
## 🛠️ Stack Tecnológico

**Framework ML:** Kedro

**Orquestación:** Apache Airflow

**Versionado:** DVC + Git

**Contenedores:** Docker + Docker Compose

**Machine Learning:** Scikit-learn, XGBoost, LightGBM

**Análisis No Supervisado:** UMAP, HDBSCAN, SHAP, Plotly

**Visualización:** Matplotlib, Seaborn, Plotly

---
## 🧪 Cómo probar tu proyecto Kedro

Revisa el archivo `tests/test_run.py` para ver ejemplos de cómo escribir tus pruebas. Puedes ejecutarlas con:

```
pytest
```

Puedes configurar el umbral de cobertura de pruebas en el archivo `pyproject.toml`, en la sección `[tool.coverage.report]`.

---

## 📚 Dependencias del proyecto

Para ver y actualizar las dependencias de el proyecto, usa el archivo `requirements.txt`.

Instálalas con:

```
pip install -r requirements.txt
```

---

## 📓 Cómo trabajar con Kedro y notebooks

> 💡 Al usar `kedro jupyter` o `kedro ipython`, tendrás acceso automático a las siguientes variables en tu notebook: `context`, `session`, `catalog` y `pipelines`.

Jupyter, JupyterLab e IPython ya están incluidos por defecto en los requerimientos del proyecto. Una vez que ejecutes:

```
pip install -r requirements.txt
```

No necesitas pasos adicionales.

### Usar Jupyter

Instala Jupyter si aún no lo tienes:

```
pip install jupyter
```

Inicia un servidor local de notebooks:

```
kedro jupyter notebook
```

### Usar JupyterLab

Instálalo con:

```
pip install jupyterlab
```

Y luego ejecútalo con:

```
kedro jupyter lab
```

### Usar IPython

Si prefieres iniciar una sesión interactiva con IPython:

```
kedro ipython
```

---

## 🚫 Cómo ignorar las salidas de celdas de los notebooks en git

Para eliminar automáticamente el contenido de las celdas de salida antes de hacer commits a git, puedes usar herramientas como `nbstripout`.

Por ejemplo, puedes añadir un *hook* con:

```
nbstripout --install
```

> ⚠️ Las salidas de tus celdas se mantendrán localmente.

---

## 📦 Empaquetar tu proyecto Kedro

Consulta la documentación oficial de Kedro para más información sobre cómo generar documentación del proyecto y empaquetarlo para su distribución.

---
## 👥 Autores
**Gonzalo Gallardo**

**Alan Barria**