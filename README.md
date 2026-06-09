# FraudLens ML

Dashboard académico en Streamlit para analizar artículos científicos exportados desde Scopus sobre Machine Learning aplicado a la detección de fraudes financieros en la industria bancaria.

## Pregunta de investigación

¿Cómo contribuye Machine Learning a la detección de fraudes financieros en la industria bancaria?

## Archivos

- `app.py`: aplicación principal de Streamlit.
- `requirements.txt`: librerías necesarias.
- `README.md`: instrucciones del proyecto.

## Configuración antes de desplegar

En `app.py`, reemplazar las siguientes variables:

```python
DATA_URL = "PEGAR_AQUI_URL_RAW_DEL_CSV"
GITHUB_URL = "PEGAR_AQUI_URL_GITHUB"
COLAB_URL = "PEGAR_AQUI_URL_COLAB"
NRC = "PEGAR_AQUI_NRC"
GRUPO = "PEGAR_AQUI_NUMERO_DE_GRUPO"
INTEGRANTES = [
    "PEGAR_INTEGRANTE_1",
    "PEGAR_INTEGRANTE_2",
    "PEGAR_INTEGRANTE_3",
]
```

## Cómo obtener la URL RAW del CSV en GitHub

1. Subir el archivo CSV al repositorio público de GitHub.
2. Abrir el archivo CSV desde GitHub.
3. Presionar el botón `Raw`.
4. Copiar la URL del navegador.
5. Pegar esa URL en la variable `DATA_URL`.

## Despliegue en Streamlit Cloud

1. Crear un repositorio público en GitHub.
2. Subir `app.py`, `requirements.txt`, `README.md` y el CSV.
3. Entrar a Streamlit Cloud.
4. Seleccionar el repositorio.
5. Elegir `app.py` como archivo principal.
6. Presionar Deploy.

## Visualizaciones incluidas

- Publicaciones por año.
- Top 10 artículos más citados.
- Distribución por tipo de documento.
- Top 20 palabras frecuentes en títulos y abstracts.
- Histograma de citas.
- Boxplot de citas.
- Nube de palabras de abstracts.
- Tabla interactiva de artículos.
