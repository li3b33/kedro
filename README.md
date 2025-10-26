# OlympicsKedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Dataset kaggle
https://www.kaggle.com/datasets/the-guardian/olympic-games

## Video precentacion
https://drive.google.com/drive/folders/1vNNrXJhd7qrNee-4m86045C1FjAWlilg?usp=sharing

## 🐍 Crear entorno virtual

Antes de comenzar, crea un entorno virtual llamado `venv` y actívalo:

```bash
python -m venv venv
source venv/bin/activate  #En Linux
venv\Scripts\activate  #En Windows
```

Una vez activado, puedes instalar las dependencias como se indica más abajo.

---

## 📝 Descripción general

Este es tu nuevo proyecto Kedro, que fue generado utilizando **Kedro 1.0.0**.

Consulta la [documentación oficial de Kedro](https://docs.kedro.org/) para comenzar.

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
