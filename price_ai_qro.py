import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. CARGA + PREPARACIÓN DE DATOS (CON INPC)
# =========================

@st.cache_data
def load_and_prepare_data():
    """
    Carga:
      - sf_ventas_qro.csv (ventas históricas)
      - inpc_mexico.csv (INPC oficial por año/mes, columnas: anio, mes, inpc)

    Ajusta todos los precios a pesos constantes usando INPC,
    y calcula:
      - precio_cerrado_real (ajustado a pesos del último periodo)
      - precio_m2 (sobre el precio ajustado)
      - crecimiento histórico por colonia (CAGR)
      - nivel relativo de precio/m² por colonia vs promedio global
    """
    # --- 1) Cargar ventas ---
    raw_path = "sf_ventas_qro.csv"
    df = pd.read_csv(raw_path)

    # Fechas
    for col in ["fecha_creacion", "fecha_firma"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        else:
            df[col] = pd.NaT

    # Año / mes de firma
    if "anio_firma" not in df.columns:
        df["anio_firma"] = df["fecha_firma"].dt.year
    if "mes_firma" not in df.columns:
        df["mes_firma"] = df["fecha_firma"].dt.month

    if "dias_creacion_a_firma" not in df.columns:
        df["dias_creacion_a_firma"] = (
            df["fecha_firma"] - df["fecha_creacion"]
        ).dt.days

    # Filtrar registros válidos básicos
    mask_valid = (
        df["precio_cerrado"].notna()
        & (df["precio_cerrado"] > 0)
        & df["m2_interiores"].notna()
        & (df["m2_interiores"] > 0)
    )
    df = df[mask_valid].copy()

    # --- 2) Cargar INPC y empatar por año/mes ---
    inpc_df = pd.read_csv("inpc_mexico.csv", encoding="utf-8")

    # Normalizar nombres de columnas por si vienen con mayúsculas/espacios
    inpc_df.columns = [c.strip().lower() for c in inpc_df.columns]

    # Esperamos columnas: anio, mes, inpc
    required_cols = {"anio", "mes", "inpc"}
    if not required_cols.issubset(set(inpc_df.columns)):
        raise ValueError(f"El CSV de INPC debe contener las columnas: {required_cols}")

    inpc_df["anio"] = inpc_df["anio"].astype(int)
    inpc_df["mes"] = inpc_df["mes"].astype(int)

    # Merge ventas + INPC según año y mes de firma
    df = df.merge(
        inpc_df[["anio", "mes", "inpc"]],
        left_on=["anio_firma", "mes_firma"],
        right_on=["anio", "mes"],
        how="left"
    )

    # Eliminar registros sin INPC (por seguridad)
    df = df[df["inpc"].notna()].copy()

    # --- 3) Ajustar precios a pesos constantes usando INPC ---
    # INPC de referencia = el más reciente de la serie (máximo)
    inpc_ref = inpc_df["inpc"].max()

    # Precio real ajustado a pesos del periodo de referencia
    df["precio_cerrado_real"] = df["precio_cerrado"] * (inpc_ref / df["inpc"])

    # Recalcular precio_m2 con el precio ajustado
    df["precio_m2"] = df["precio_cerrado_real"] / df["m2_interiores"]

    # Eliminar outliers extremos en precio/m2 (1% y 99%)
    q1, q99 = df["precio_m2"].quantile([0.01, 0.99])
    df = df[(df["precio_m2"] >= q1) & (df["precio_m2"] <= q99)].copy()

    # --- 4) Niveles de precio/m² por colonia ---
    med_price_m2_global = df["precio_m2"].median()
    med_price_m2_colonia = (
        df.groupby("colonia")["precio_m2"]
        .median()
        .to_dict()
    )

    # --- 5) Crecimiento histórico por colonia (CAGR en % anual) ---
    growth_colonia = {}
    trend = (
        df.groupby(["colonia", "anio_firma"])["precio_m2"]
        .median()
        .reset_index()
    )

    for col_name, grp in trend.groupby("colonia"):
        grp = grp.dropna(subset=["anio_firma", "precio_m2"]).sort_values("anio_firma")
        if grp["anio_firma"].nunique() >= 2:
            first = grp["precio_m2"].iloc[0]
            last = grp["precio_m2"].iloc[-1]
            years = grp["anio_firma"].iloc[-1] - grp["anio_firma"].iloc[0]
            if first > 0 and years > 0:
                cagr = (last / first) ** (1 / years) - 1
                growth_colonia[col_name] = cagr * 100.0  # en %

    default_growth = float(
        np.median(list(growth_colonia.values()))
    ) if growth_colonia else 5.0

    latest_year = int(df["anio_firma"].max())

    return (
        df,
        growth_colonia,
        default_growth,
        latest_year,
        med_price_m2_colonia,
        med_price_m2_global,
    )

# =========================
# 2. MODELO
# =========================

@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Entrena un RandomForestRegressor usando como variable categórica la colonia.
    El modelo predice precio_cerrado_real (precio ajustado por INPC).
    """
    numeric_features = [
        "m2_interiores",
        "m2_totales",
        "m2_jardin",
        "m2_terraza",
        "estacionamientos",
        "descuento_pct",
        "anio_firma",
        "mes_firma",
        "dias_creacion_a_firma",
    ]
    cat_features = ["colonia"]

    X_num = df[numeric_features].fillna(0)
    X_cat = pd.get_dummies(df[cat_features].astype(str), prefix=cat_features)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["precio_cerrado_real"]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def build_input_row(
    colonia,
    m2_int,
    m2_tot,
    m2_jard,
    m2_terr,
    estac,
    descuento_pct,
    anio_ref,
    mes_ref,
    dias_medios,
    feature_columns,
):
    """Construye una sola fila de input con el mismo esquema que el training."""
    numeric_features = {
        "m2_interiores": m2_int,
        "m2_totales": m2_tot,
        "m2_jardin": m2_jard,
        "m2_terraza": m2_terr,
        "estacionamientos": estac,
        "descuento_pct": descuento_pct,
        "anio_firma": anio_ref,
        "mes_firma": mes_ref,
        "dias_creacion_a_firma": dias_medios,
    }

    row = pd.DataFrame([numeric_features])

    dummy = pd.get_dummies(
        pd.Series([colonia], name="colonia").astype(str),
        prefix=["colonia"]
    )

    for col in feature_columns:
        if col.startswith("colonia_"):
            if col in dummy.columns:
                row[col] = dummy[col].iloc[0]
            else:
                row[col] = 0

    # Asegurar todas las columnas
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_columns]
    return row


# =========================
# 3. APP STREAMLIT – PRECIOS ACTUALES Y FUTUROS
# =========================

def main():
    st.set_page_config(
        page_title="AI Pricing Engine – Querétaro",
        layout="wide"
    )

    st.title("🏙️ AI Pricing Engine – Querétaro")
    st.caption(
        "Motor de pricing entrenado con ventas reales desde 2013, "
        "ajustadas por INPC, para estimar precios actuales y futuros por zona/producto."
    )

    (
        df,
        growth_colonia,
        default_growth,
        latest_year,
        med_price_m2_colonia,
        med_price_m2_global,
    ) = load_and_prepare_data()

    model, feature_columns = train_model(df)

    # ================= SIDEBAR =================
    st.sidebar.header("🎯 Producto objetivo (nuevo proyecto)")

    colonia = st.sidebar.selectbox(
        "Colonia / zona de referencia",
        sorted(df["colonia"].dropna().astype(str).unique())
    )

    st.sidebar.markdown("### 🧱 Características físicas")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        m2_int = st.number_input(
            "m² interiores",
            min_value=40.0, max_value=600.0,
            value=120.0, step=5.0
        )
        m2_jard = st.number_input(
            "m² jardín",
            min_value=0.0, max_value=300.0,
            value=0.0, step=5.0
        )
    with col2:
        m2_terr = st.number_input(
            "m² terraza",
            min_value=0.0, max_value=200.0,
            value=8.0, step=2.0
        )
        estac = st.number_input(
            "Cajones de estacionamiento",
            min_value=0, max_value=6,
            value=2, step=1
        )

    # m² totales calculados automáticamente
    m2_tot = m2_int + m2_jard + m2_terr

    st.sidebar.markdown(
        f"**m² totales (calculado):** `{m2_tot:,.1f} m²`"
    )

    descuento_pct = st.sidebar.number_input(
        "Descuento objetivo (%)",
        min_value=0.0, max_value=20.0,
        value=5.0, step=0.5
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Proyección de precios (histórico + inflación futura)")

    # Crecimiento histórico por colonia (ya en precios reales)
    hist_growth = growth_colonia.get(colonia, default_growth)

    inflacion = st.sidebar.slider(
        "Inflación esperada anual futura (%)",
        0.0, 12.0, 4.0, step=0.5
    )

    peso_hist = st.sidebar.slider(
        "Peso del histórico de la colonia (%)",
        0, 100, 70, step=5,
        help="Ejemplo: 70% histórico de la colonia, 30% inflación esperada futura."
    )

    horizonte_meses = st.sidebar.slider(
        "Horizonte de proyección (meses)",
        0, 36, 12, step=3
    )

    st.sidebar.markdown("---")
    precio_objetivo = st.sidebar.number_input(
        "Tu precio de lista (MXN)",
        min_value=500_000.0,
        max_value=40_000_000.0,
        value=3_500_000.0,
        step=50_000.0
    )

    # ================= PREDICCIÓN =================

    anio_ref = latest_year
    mes_ref = int(df["mes_firma"].dropna().mode().iloc[0])
    dias_medios = int(df["dias_creacion_a_firma"].dropna().median())

    input_row = build_input_row(
        colonia=colonia,
        m2_int=m2_int,
        m2_tot=m2_tot,
        m2_jard=m2_jard,
        m2_terr=m2_terr,
        estac=estac,
        descuento_pct=descuento_pct,
        anio_ref=anio_ref,
        mes_ref=mes_ref,
        dias_medios=dias_medios,
        feature_columns=feature_columns,
    )

    # Predicción base del modelo (precio real actual)
    precio_base_real = float(model.predict(input_row)[0])

    # Ajuste por nivel de precio/m² de la colonia vs el promedio global
    med_colonia = med_price_m2_colonia.get(colonia, med_price_m2_global)
    factor_zona = med_colonia / med_price_m2_global if med_price_m2_global > 0 else 1.0

    precio_hoy = precio_base_real * factor_zona
    precio_hoy_min = precio_hoy * 0.95
    precio_hoy_max = precio_hoy * 1.05
    precio_m2_hoy = precio_hoy / m2_int if m2_int > 0 else 0

    # Mezcla: histórico colonia (ya real) + inflación futura
    peso_hist_f = peso_hist / 100.0
    g_efectivo = hist_growth * peso_hist_f + inflacion * (1 - peso_hist_f)

    factor_tiempo = (1 + g_efectivo / 100.0) ** (horizonte_meses / 12.0)
    precio_futuro = precio_hoy * factor_tiempo
    precio_futuro_min = precio_hoy_min * factor_tiempo
    precio_futuro_max = precio_hoy_max * factor_tiempo
    precio_m2_fut = precio_futuro / m2_int if m2_int > 0 else 0

    delta_abs_hoy = precio_objetivo - precio_hoy
    delta_pct_hoy = (delta_abs_hoy / precio_hoy * 100) if precio_hoy > 0 else 0

    # ================= LAYOUT: TABS =================

    tab_resumen, tab_detalle, tab_hist = st.tabs(
        ["📊 Resumen ejecutivo", "📉 Detalle del escenario", "📈 Historial por colonia"]
    )

    # ---- TAB 1: RESUMEN ----
    with tab_resumen:
        st.subheader("📊 Recomendación de precio – actual y futuro")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Precio recomendado HOY", f"${precio_hoy:,.0f} MXN")
            st.metric("Precio mínimo HOY", f"${precio_hoy_min:,.0f} MXN")
        with c2:
            st.metric(
                f"Precio recomendado en {horizonte_meses} meses",
                f"${precio_futuro:,.0f} MXN"
            )
            st.metric(
                f"Precio máximo en {horizonte_meses} meses",
                f"${precio_futuro_max:,.0f} MXN"
            )
        with c3:
            st.metric("Precio HOY por m²", f"${precio_m2_hoy:,.0f} MXN/m²")
            st.metric(
                f"Precio futuro por m²",
                f"${precio_m2_fut:,.0f} MXN/m²"
            )

        st.markdown("---")

        if abs(delta_pct_hoy) < 3:
            st.success(
                "Tu precio objetivo está muy alineado con el precio recomendado HOY "
                "por el modelo."
            )
        elif delta_pct_hoy > 3:
            st.warning(
                "Tu precio objetivo está por encima del precio recomendado HOY. "
                "Puede implicar mayor tiempo en mercado o necesidad de descuento."
            )
        else:
            st.info(
                "Tu precio objetivo está por debajo del precio recomendado HOY. "
                "Puede favorecer velocidad de venta o indicar oportunidad de capturar más valor."
            )

        st.markdown(
            f"""
            - El modelo está entrenado con **{len(df):,} operaciones reales**, 
              todas ajustadas a precios constantes usando INPC oficial.  
            - Multiplicador de zona para **{colonia}** (precio/m² colonia / global): **{factor_zona:,.2f}x**.  
            - Crecimiento histórico estimado (real) para **{colonia}**: **{hist_growth:.1f}% anual**.  
            - Inflación futura esperada: **{inflacion:.1f}% anual**.  
            - Tasa efectiva usada (histórico + inflación futura): **{g_efectivo:.1f}% anual**.
            """
        )

    # ---- TAB 2: DETALLE ----
    with tab_detalle:
        st.subheader("📉 Detalle del producto y supuestos de proyección")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Configuración física del producto")
            st.write(f"- Colonia / zona de referencia: **{colonia}**")
            st.write(f"- m² interiores: **{m2_int:.1f} m²**")
            st.write(f"- m² totales (calculado): **{m2_tot:.1f} m²**")
            st.write(f"- m² terraza: **{m2_terr:.1f} m²**")
            st.write(f"- m² jardín: **{m2_jard:.1f} m²**")
            st.write(f"- Cajones de estacionamiento: **{estac}**")

        with col_der:
            st.markdown("#### Supuestos comerciales y de mercado")
            st.write(f"- Descuento objetivo vs lista: **{descuento_pct:.1f}%**")
            st.write(f"- Horizonte de proyección: **{horizonte_meses} meses**")
            st.write(f"- Crecimiento histórico colonia (real): **{hist_growth:.1f}% anual**")
            st.write(f"- Inflación futura esperada: **{inflacion:.1f}% anual**")
            st.write(f"- Peso histórico colonia: **{peso_hist}%**")
            st.write(f"- Tasa efectiva usada: **{g_efectivo:.1f}% anual**")
            st.write(f"- Multiplicador de zona (precio/m² colonia / global): **{factor_zona:,.2f}x**")
            st.write(f"- Precio recomendado HOY: **${precio_hoy:,.0f} MXN**")
            st.write(
                f"- Precio recomendado en {horizonte_meses} meses: "
                f"**${precio_futuro:,.0f} MXN**"
            )
            st.write(f"- Tu precio de lista objetivo HOY: **${precio_objetivo:,.0f} MXN**")
            st.write(
                f"- Diferencia vs precio IA HOY: "
                f"**${delta_abs_hoy:,.0f} MXN ({delta_pct_hoy:,.2f}%)**"
            )

      # ---- TAB 3: HISTÓRICO ----
    with tab_hist:
        st.subheader("📈 Historial de precios por m² en la colonia (precios reales)")

        df_col = df[df["colonia"] == colonia].copy()

        if not df_col.empty:
            st.write(
                f"Historial de ventas para **{colonia}** "
                f"({len(df_col)} operaciones depuradas, ajustadas por INPC)."
            )

            # -------- TABLA FORMATEADA COMO MONEDA --------
            df_hist = df_col[
                ["anio_firma", "mes_firma", "m2_interiores",
                 "precio_cerrado_real", "precio_m2"]
            ].sort_values(["anio_firma", "mes_firma"]).copy()

            df_hist["precio_cerrado_real"] = df_hist["precio_cerrado_real"].apply(
                lambda x: f"${x:,.2f}"
            )
            df_hist["precio_m2"] = df_hist["precio_m2"].apply(
                lambda x: f"${x:,.2f}"
            )

            st.dataframe(df_hist)

            # -------- GRÁFICA --------
            st.markdown("#### Evolución histórica de precio/m² real (mediana anual)")
            pivot = (
                df_col.groupby("anio_firma")["precio_m2"]
                .median()
                .reset_index()
                .set_index("anio_firma")
            )
            st.line_chart(pivot)

        else:
            st.info("No hay historial suficiente para esta colonia en la base.")



if __name__ == "__main__":
    main()
