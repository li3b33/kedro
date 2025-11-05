# OlympicsKedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Dataset kaggle
https://www.kaggle.com/datasets/the-guardian/olympic-games

## Video presentación Ev2 
https://drive.google.com/file/d/1ufLL5GsWMHaNYclvSCpVqR4Nqa5dyV99/view?usp=sharing
[![Video Explicativo OlympicsKedro](https://img.shields.io/badge/Video%20Explicativo-OlympicsKedro-blue?logo=google-drive)](https://drive.google.com/file/d/10TGaQiC0rRztdoMYyrjEVfpetHDd8PZD/view?usp=sharing)


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

Kedro → para estructurar y ejecutar pipelines de datos reproducibles

DVC → para versionar datasets y modelos

Airflow → para orquestar la ejecución automatizada de pipelines

Docker → para desplegar y ejecutar todo el ecosistema en contenedores

## El proyecto incluye pipelines de:

Data Engineering (preprocesamiento y limpieza)

Classification (5 modelos de clasificación)

Regression (modelos de predicción continua)

Reporting (generación de métricas y resultados)

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
kedro run --pipeline=data_engineering
kedro run --pipeline=classification
kedro run --pipeline=regression
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

(https://www.docker.com)

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
