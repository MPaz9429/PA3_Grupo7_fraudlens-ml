# app.py
# Dashboard académico para PA3 - Fundamentos de Machine Learning
# Tema: Machine Learning aplicado a la detección de fraudes financieros en la industria bancaria

import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

# =========================================================
# 1. CONFIGURACIÓN EDITABLE
# =========================================================

# Reemplaza esta URL por la URL RAW de tu CSV en GitHub.
DATA_URL = "https://raw.githubusercontent.com/MPaz9429/PA3_Grupo7_fraudlens-ml/main/scopus_ML_Banking.csv"

# Enlaces del proyecto.
GITHUB_URL = "https://github.com/MPaz9429/PA3_Grupo7_fraudlens-ml"
COLAB_URL = "https://colab.research.google.com/drive/1KxqHoyITv6_AubWMKqgRQ8DwgtoUpDuk?usp=sharing"

# Información académica.
NRC = "PEGAR_AQUI_NRC"
GRUPO = "PEGAR_AQUI_NUMERO_DE_GRUPO"
INTEGRANTES = [
    "Mercedes",
    "Leon",
    "Alejandra",
]

PREGUNTA_INVESTIGACION = "¿Cómo contribuye Machine Learning a la detección de fraudes financieros en la industria bancaria?"
KEYWORDS = ["Machine Learning", "Fraud Detection", "Financial Fraud", "Banking"]

# =========================================================
# 2. CONFIGURACIÓN GENERAL DE STREAMLIT
# =========================================================

st.set_page_config(
    page_title="FraudLens ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos visuales del dashboard.
st.markdown(
    """
    <style>
    .main {
        background-color: #F7FAFC;
    }
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: linear-gradient(135deg, #081C3A 0%, #0B3A66 45%, #0EA5E9 100%);
        padding: 2.2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0px 10px 28px rgba(8, 28, 58, 0.25);
    }
    .hero h1 {
        font-size: 2.35rem;
        margin-bottom: 0.35rem;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.05rem;
        color: #E5F6FF;
        max-width: 1100px;
    }
    .section-card {
        background-color: white;
        padding: 1.2rem 1.35rem;
        border-radius: 18px;
        border: 1px solid #E6EEF6;
        box-shadow: 0px 4px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .interpretation {
        background-color: #EFF8FF;
        border-left: 5px solid #0EA5E9;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        color: #0F2A43;
        margin-top: 0.55rem;
        margin-bottom: 0.8rem;
    }
    .footer {
        background-color: #081C3A;
        color: white;
        padding: 1.3rem;
        border-radius: 18px;
        margin-top: 1rem;
    }
    .small-muted {
        color: #64748B;
        font-size: 0.92rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. FUNCIONES DE APOYO
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """Carga el CSV desde una URL RAW de GitHub."""
    return pd.read_csv(url)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres esperados de columnas Scopus sin romper el CSV original."""
    rename_map = {}
    for col in df.columns:
        clean = col.strip().lower()
        if clean in ["title", "document title"]:
            rename_map[col] = "Title"
        elif clean in ["authors", "author"]:
            rename_map[col] = "Authors"
        elif clean in ["year", "publication year"]:
            rename_map[col] = "Year"
        elif clean in ["cited by", "cited by count", "citations"]:
            rename_map[col] = "Cited by"
        elif clean in ["abstract", "description"]:
            rename_map[col] = "Abstract"
        elif clean in ["author keywords", "keywords"]:
            rename_map[col] = "Author Keywords"
        elif clean in ["source title", "source", "journal"]:
            rename_map[col] = "Source title"
        elif clean in ["document type", "type"]:
            rename_map[col] = "Document Type"
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Devuelve columnas existentes para evitar errores si una columna no aparece."""
    return [c for c in cols if c in df.columns]


def shorten_text(text: str, max_len: int = 70) -> str:
    """Acorta títulos extensos para que los gráficos sean legibles."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def get_words(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Extrae palabras frecuentes desde títulos y abstracts."""
    custom_stopwords = {
        "study", "paper", "result", "results", "method", "methods", "data", "model",
        "models", "analysis", "using", "based", "approach", "proposed", "research",
        "system", "systems", "different", "also", "used", "provide", "shows", "show",
        "financial", "fraud", "detection", "machine", "learning"
    }
    stop_words = set(STOPWORDS).union(custom_stopwords)
    text = " ".join(series.dropna().astype(str)).lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    filtered_words = [w for w in words if w not in stop_words]
    counts = Counter(filtered_words).most_common(top_n)
    return pd.DataFrame(counts, columns=["Palabra", "Frecuencia"])


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convierte a numérico reemplazando errores por cero."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def interpretation_box(text: str):
    st.markdown(f"<div class='interpretation'>💡 {text}</div>", unsafe_allow_html=True)


def make_wordcloud(text: str):
    custom_stopwords = set(STOPWORDS).union({
        "study", "paper", "result", "results", "method", "methods", "data", "model",
        "models", "analysis", "using", "based", "approach", "proposed", "research",
        "financial", "fraud", "detection", "machine", "learning"
    })
    wc = WordCloud(
        width=1400,
        height=650,
        background_color="white",
        stopwords=custom_stopwords,
        collocations=False,
        max_words=90,
        contour_width=1,
        contour_color="#0EA5E9",
    ).generate(text)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# =========================================================
# 4. ENCABEZADO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ FraudLens ML: Inteligencia Artificial para la Detección de Fraudes Financieros Bancarios</h1>
        <p>Dashboard académico basado en artículos científicos indexados en Scopus para analizar cómo Machine Learning contribuye a la detección de fraudes financieros dentro del sector bancario.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 5. CARGA DEL DATASET
# =========================================================

if DATA_URL == "PEGAR_AQUI_URL_RAW_DEL_CSV":
    st.warning("⚠️ Reemplaza DATA_URL por la URL RAW del CSV alojado en GitHub para visualizar el dashboard.")
    st.stop()

try:
    df = load_data(DATA_URL)
    df = normalize_columns(df)
except Exception as e:
    st.error("No se pudo cargar el CSV desde GitHub. Verifica que la URL sea RAW y que el repositorio sea público.")
    st.exception(e)
    st.stop()

# Validaciones y limpieza base.
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
if "Cited by" in df.columns:
    df["Cited by"] = safe_numeric(df["Cited by"])
else:
    df["Cited by"] = 0

for col in ["Title", "Authors", "Abstract", "Author Keywords", "Source title", "Document Type"]:
    if col not in df.columns:
        df[col] = "No disponible"

# =========================================================
# 6. CONTEXTO DEL PROYECTO
# =========================================================

with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📌 Contexto del proyecto")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Pregunta de investigación:** {PREGUNTA_INVESTIGACION}")
        st.markdown("**Keywords:** " + ", ".join([f"`{k}`" for k in KEYWORDS]))
        st.markdown("**Fuente del dataset:** Scopus")
    with col2:
        st.markdown(f"**Repositorio GitHub:** [Abrir enlace]({GITHUB_URL})")
        st.markdown(f"**Google Colab base:** [Abrir enlace]({COLAB_URL})")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 7. SIDEBAR - FILTROS
# =========================================================

st.sidebar.header("🔎 Filtros interactivos")

filtered = df.copy()

if "Year" in filtered.columns and filtered["Year"].notna().any():
    years = sorted(filtered["Year"].dropna().astype(int).unique().tolist())
    selected_years = st.sidebar.multiselect("Año de publicación", years, default=years)
    filtered = filtered[filtered["Year"].astype("Int64").isin(selected_years)]

if "Document Type" in filtered.columns:
    doc_types = sorted(filtered["Document Type"].dropna().astype(str).unique().tolist())
    selected_doc_types = st.sidebar.multiselect("Tipo de documento", doc_types, default=doc_types)
    filtered = filtered[filtered["Document Type"].astype(str).isin(selected_doc_types)]

if "Source title" in filtered.columns:
    sources = sorted(
        filtered["Source title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_sources = st.sidebar.multiselect(
        "Revista/Fuente",
        sources,
        default=sources
    )

    filtered = filtered[
        filtered["Source title"].astype(str).isin(selected_sources)
    ]

search_term = st.sidebar.text_input("Buscar palabra en título o abstract")
if search_term:
    mask = (
        filtered["Title"].astype(str).str.contains(search_term, case=False, na=False)
        | filtered["Abstract"].astype(str).str.contains(search_term, case=False, na=False)
    )
    filtered = filtered[mask]

# =========================================================
# 8. KPI CARDS
# =========================================================

st.subheader("📊 Indicadores principales")

if filtered.empty:
    st.warning("No hay datos disponibles con los filtros seleccionados.")
    st.stop()

total_articles = len(filtered)
year_min = int(filtered["Year"].dropna().min()) if "Year" in filtered.columns and filtered["Year"].notna().any() else "N/D"
year_max = int(filtered["Year"].dropna().max()) if "Year" in filtered.columns and filtered["Year"].notna().any() else "N/D"
total_citations = int(filtered["Cited by"].sum())
avg_citations = round(filtered["Cited by"].mean(), 2)
most_cited_row = filtered.sort_values("Cited by", ascending=False).iloc[0]
most_cited_title = shorten_text(most_cited_row.get("Title", "No disponible"), 55)
most_common_year = int(filtered["Year"].mode().iloc[0]) if "Year" in filtered.columns and filtered["Year"].notna().any() else "N/D"

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("📚 Total de artículos", total_articles)
c2.metric("🗓️ Rango de años", f"{year_min} - {year_max}")
c3.metric("🔗 Total de citas", total_citations)
c4.metric("📈 Promedio de citas", avg_citations)
c5.metric("🏆 Artículo más citado", most_cited_title)
c6.metric("⭐ Año con más publicaciones", most_common_year)

st.divider()

# =========================================================
# 9. GRÁFICOS
# =========================================================

# A. Publicaciones por año.
st.subheader("A. 📈 Publicaciones por año")
if "Year" in filtered.columns and filtered["Year"].notna().any():
    year_counts = filtered.groupby("Year").size().reset_index(name="Publicaciones").sort_values("Year")
    fig_year = px.bar(
        year_counts,
        x="Year",
        y="Publicaciones",
        text="Publicaciones",
        title="Evolución de publicaciones científicas por año",
    )
    fig_year.update_traces(textposition="outside")
    fig_year.update_layout(yaxis_title="Número de publicaciones", xaxis_title="Año")
    st.plotly_chart(fig_year, use_container_width=True)
    interpretation_box("Este gráfico permite observar la evolución temporal de la producción científica sobre Machine Learning aplicado al fraude financiero bancario. Un aumento en los años recientes puede indicar mayor interés académico por la banca digital, transacciones electrónicas y detección automatizada de riesgos.")
else:
    st.info("No se encontró una columna de año válida.")

# B. Top artículos más citados.
st.subheader("B. 🏆 Top 10 artículos más citados")
top_cited = filtered.sort_values("Cited by", ascending=False).head(10).copy()
top_cited["Título abreviado"] = top_cited["Title"].apply(lambda x: shorten_text(x, 75))
fig_cited = px.bar(
    top_cited.sort_values("Cited by", ascending=True),
    x="Cited by",
    y="Título abreviado",
    orientation="h",
    title="Artículos con mayor impacto académico según número de citas",
    text="Cited by",
)
fig_cited.update_layout(xaxis_title="Número de citas", yaxis_title="Artículo")
st.plotly_chart(fig_cited, use_container_width=True)
interpretation_box("Los artículos más citados representan investigaciones con mayor impacto académico. En el contexto del fraude financiero, suelen concentrarse en modelos de clasificación, detección de anomalías y prevención de fraudes en transacciones bancarias.")

# C. Distribución por tipo de documento.
st.subheader("C. 📄 Distribución por tipo de documento")
doc_counts = filtered["Document Type"].value_counts().reset_index()
doc_counts.columns = ["Tipo de documento", "Cantidad"]
fig_doc = px.pie(
    doc_counts,
    names="Tipo de documento",
    values="Cantidad",
    title="Composición del dataset según tipo de documento",
    hole=0.38,
)
st.plotly_chart(fig_doc, use_container_width=True)
interpretation_box("Esta distribución permite identificar si el dataset está compuesto principalmente por artículos científicos, conferencias, revisiones u otros tipos de documentos. Para la PA3, los artículos científicos son especialmente relevantes porque se alinean mejor con la rúbrica de Scopus.")

# D. Palabras frecuentes.
st.subheader("D. 🔤 Top 20 palabras frecuentes en títulos y abstracts")
text_cols = require_columns(filtered, ["Title", "Abstract"])
combined_text = filtered[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else pd.Series(dtype=str)
word_freq = get_words(combined_text, top_n=20)
if not word_freq.empty:
    fig_words = px.bar(
        word_freq.sort_values("Frecuencia", ascending=True),
        x="Frecuencia",
        y="Palabra",
        orientation="h",
        title="Conceptos predominantes en títulos y abstracts",
        text="Frecuencia",
    )
    fig_words.update_layout(xaxis_title="Frecuencia", yaxis_title="Palabra")
    st.plotly_chart(fig_words, use_container_width=True)
    interpretation_box("Las palabras frecuentes ayudan a reconocer los conceptos predominantes de la literatura científica, como algoritmos, transacciones, detección de anomalías, tarjetas de crédito y seguridad bancaria.")
else:
    st.info("No se pudo generar frecuencia de palabras con los datos disponibles.")

# E. Histograma de citas.
st.subheader("E. 📊 Histograma de citas")
st.markdown("**Pregunta:** ¿Cómo se distribuye el impacto académico de las investigaciones sobre detección de fraudes financieros mediante Machine Learning?")
fig_hist = px.histogram(
    filtered,
    x="Cited by",
    nbins=12,
    title="Distribución del impacto académico medido por citas",
)
fig_hist.update_layout(xaxis_title="Número de citas", yaxis_title="Cantidad de artículos")
st.plotly_chart(fig_hist, use_container_width=True)
interpretation_box("El histograma permite observar si la mayoría de artículos tiene pocas citas o si existen grupos con alto impacto. En investigaciones recientes es normal encontrar varios documentos con pocas citas debido a su reciente publicación.")

# F. Boxplot de citas.
st.subheader("F. 📦 Boxplot de citas")
st.markdown("**Pregunta:** ¿La producción científica está concentrada en pocos artículos altamente citados o distribuida de manera homogénea?")
fig_box = px.box(
    filtered,
    y="Cited by",
    points="all",
    title="Concentración del impacto académico por número de citas",
)
fig_box.update_layout(yaxis_title="Número de citas")
st.plotly_chart(fig_box, use_container_width=True)
interpretation_box("El boxplot muestra la mediana, los cuartiles y los valores atípicos. Si aparecen puntos alejados del resto, puede interpretarse que el impacto académico está concentrado en pocos artículos altamente citados.")

# G. Nube de palabras.
st.subheader("G. ☁️ Nube de palabras en abstracts")
abstract_text = " ".join(filtered["Abstract"].dropna().astype(str))
if len(abstract_text.strip()) > 0:
    fig_wc = make_wordcloud(abstract_text)
    st.pyplot(fig_wc)
    interpretation_box("La nube de palabras resume visualmente los términos más representativos de los abstracts. Esto ayuda a identificar los enfoques principales del campo, como fraude, transacciones, algoritmos, anomalías, banca digital e inteligencia artificial.")
else:
    st.info("No hay abstracts suficientes para generar la nube de palabras.")

# =========================================================
# 10. TABLA INTERACTIVA
# =========================================================

st.divider()
st.subheader("🔍 Tabla interactiva de artículos")
cols_to_show = require_columns(
    filtered,
    ["Title", "Authors", "Year", "Cited by", "Document Type", "Source title", "Abstract"]
)
st.dataframe(filtered[cols_to_show], use_container_width=True, hide_index=True)

with st.expander("📥 Ver columnas disponibles en el CSV"):
    st.write(list(df.columns))

# =========================================================
# 11. CIERRE ACADÉMICO Y FOOTER
# =========================================================

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🧠 Interpretación general")
st.write(
    "El dashboard permite observar cómo la literatura científica analiza el uso de Machine Learning en la detección de fraudes financieros dentro de la industria bancaria. "
    "A partir de los artículos de Scopus se identifican tendencias de publicación, documentos con mayor impacto, tipos de producción científica y conceptos frecuentes en abstracts. "
    "En conjunto, estos elementos ayudan a responder la pregunta de investigación al evidenciar que Machine Learning se utiliza principalmente para detectar patrones anómalos, clasificar transacciones sospechosas y fortalecer la seguridad financiera digital."
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="footer">
        <h3>📘 Información del Proyecto</h3>
        <p><b>Curso:</b> Fundamentos de Machine Learning</p>
        <p><b>Actividad:</b> PA3 – Dashboard Científico con Scopus y Streamlit</p>
        <p><b>NRC:</b> {NRC}</p>
        <p><b>Grupo:</b> {GRUPO}</p>
        <p><b>Integrantes:</b></p>
        <ul>
            <li>{INTEGRANTES[0]}</li>
            <li>{INTEGRANTES[1]}</li>
            <li>{INTEGRANTES[2]}</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
