# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO
import re

st.set_page_config(page_title="Proyección de Ventas", layout="wide")
st.title("📊 Proyección de Ventas con Variables Macroeconómicas")

# ---------- utilidades ----------
def normalize_cols(cols):
    rep = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    out = []
    for c in cols:
        c2 = str(c).translate(rep)
        c2 = re.sub(r"\s+", "", c2)  # sin espacios
        out.append(c2)
    return out

def to_numeric_df(df):
    # quita %,$,comas y convierte a numerico
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def scale_to_percentage_points(df, exclude=("Año","Ano","Year")):
    # si una columna (no excluida) tiene valores en 0..1, multiplica *100
    for c in df.columns:
        if c in exclude: 
            continue
        col = df[c]
        if col.dropna().abs().max() <= 1.5:  # dato viene como proporción
            df[c] = col * 100.0
    return df

# ---------- carga de datos ----------
st.subheader("📂 Cargar datos históricos")
st.markdown(
    "Puedes subir **CSV o Excel**. La app acepta **variables en filas o en columnas**. "
    "Variables requeridas: **PIB, Desempleo, TipoCambioPct, Inflacion, Ventas**."
)
up = st.file_uploader("Archivo histórico (CSV/XLSX)", type=["csv","xlsx"])

if not up:
    st.info("Sube primero tu archivo histórico para continuar.")
    st.stop()

# Lee y normaliza
if up.name.endswith(".csv"):
    raw = pd.read_csv(up, header=0)
else:
    raw = pd.read_excel(up, header=0)

raw.columns = normalize_cols(raw.columns)
raw = raw.loc[:, ~raw.columns.duplicated()]  # elimina duplicados exactos

# ¿Vienen años como columnas y variables en filas? (tu caso de Excel)
need_transpose = False
if "Ventas" not in raw.columns and "Ventas" in normalize_cols(list(raw.iloc[:,0])):
    raw = raw.set_index(raw.columns[0])
    need_transpose = True

if need_transpose:
    df = raw.transpose().reset_index().rename(columns={"index":"Año"})
else:
    df = raw.copy()

df.columns = normalize_cols(df.columns)
# Si no hay columna Año, intenta inferirla del índice
if "Año" not in df.columns and "Ano" not in df.columns and "Year" not in df.columns:
    df.insert(0, "Año", range(1, len(df)+1))

# convierte a numérico y escala a puntos porcentuales
df = to_numeric_df(df)
df = scale_to_percentage_points(df)

st.caption("Datos históricos (tras limpieza/normalización). Todos los porcentajes están en **puntos porcentuales (0–100)**.")
st.dataframe(df, use_container_width=True)

# Validación de columnas requeridas
req = ["PIB","Desempleo","TipoCambioPct","Inflacion","Ventas"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}. Renombra en tu archivo y vuelve a subir.")
    st.stop()

# ---------- regresiones simples ----------
X_vars = ["PIB","Desempleo","TipoCambioPct","Inflacion"]
y = df["Ventas"].astype(float).values

pend, inter, r2 = {}, {}, {}
corr = {}

for var in X_vars:
    x = df[[var]].astype(float).values
    model = LinearRegression().fit(x, y)
    pend[var] = float(model.coef_[0])
    inter[var] = float(model.intercept_)
    r2[var]   = float(model.score(x, y))
    corr[var] = float(pd.Series(df[var]).corr(pd.Series(df["Ventas"])))

res = pd.DataFrame({
    "Variable": X_vars,
    "Correlacion": [corr[v] for v in X_vars],
    "Pendiente (β)": [pend[v] for v in X_vars],
    "Interseccion (α)": [inter[v] for v in X_vars],
    "R²": [r2[v] for v in X_vars]
})
st.subheader("📈 Correlaciones y regresiones simples")
st.dataframe(res, use_container_width=True)

# ---------- sidebar: pronósticos ----------
st.sidebar.header("🔮 Escenarios (pronósticos macro)")
st.sidebar.caption("Ingresa valores en **puntos porcentuales** (ej. 2.5 = 2.5%).")

pib_f   = st.sidebar.number_input("Variación PIB (%)",      value=2.50)
des_f   = st.sidebar.number_input("Desempleo (%)",          value=3.90)
tc_f    = st.sidebar.number_input("Tipo de cambio (%)",     value=0.28)
infl_f  = st.sidebar.number_input("Inflación (%)",          value=4.80)

forecast = {"PIB":pib_f, "Desempleo":des_f, "TipoCambioPct":tc_f, "Inflacion":infl_f}

# ---------- pronósticos de ventas ----------
ventas_simple = {v: inter[v] + pend[v]*forecast[v] for v in X_vars}

total_r2 = sum(r2.values()) if sum(r2.values()) != 0 else 1.0
pesos = {v: r2[v]/total_r2 for v in X_vars}
ventas_ponderada = sum(ventas_simple[v]*pesos[v] for v in X_vars)

pred_tbl = pd.DataFrame({
    "Variable": X_vars,
    "Pronostico Ventas (%)": [ventas_simple[v] for v in X_vars],
    "Peso (R²)": [pesos[v] for v in X_vars]
})

st.subheader("📊 Pronóstico de Ventas con Regresiones Simples")
st.dataframe(pred_tbl.style.format({"Pronostico Ventas (%)":"{:.3f}","Peso (R²)":"{:.3f}"}), use_container_width=True)

st.subheader("⚖️ Pronóstico de Ventas (regresión múltiple ponderada por R²)")
st.metric("Ventas proyectadas (%)", f"{ventas_ponderada:.3f}")

# ---------- descarga ----------
out = BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as w:
    df.to_excel(w, sheet_name="Datos_Historicos", index=False)
    res.to_excel(w, sheet_name="Regresiones", index=False)
    pred_tbl.to_excel(w, sheet_name="Pronosticos", index=False)
st.download_button(
    "📥 Descargar resultados (Excel)",
    data=out.getvalue(),
    file_name="Resultados_Proyecciones.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)