import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import numpy as np

# =========================
# 1. CARGA + LIMPIEZA + TENDENCIAS
# =========================

@st.cache_data
def load_and_prepare_data():
    """
    Carga el CSV crudo exportado desde Salesforce
    y genera un dataset limpio + tendencias de precio/m2 por desarrollo.
    """
    raw_path = "sf_ventas_qro.csv"  # <--- Asegúrate que se llame así en el repo
    df_raw = pd.read_csv(raw_path)

    # Asegurar columnas de terraza
    for c in ["Mts 2 Terraza", "Mts2 Terraza"]:
        if c not in df_raw.columns:
            df_raw[c] = 0.0

    df_raw["terraza_m2"] = df_raw[["Mts 2 Terraza", "Mts2 Terraza"]].fillna(0).max(axis=1)

    df = pd.DataFrame({
        "desarrollo": df_raw["Desarrollo"].astype(str).str.strip(),
        "colonia": df_raw["Colonia"].astype(str).str.strip(),
        "m2_interiores": df_raw["M2 Construcción Privativa"],
        "m2_totales": df_raw["Área Privativa Total (m2)"],
        "m2_jardin": df_raw["M2 Jardín"].fillna(0),
        "m2_terraza": df_raw["terraza_m2"].fillna(0),
        "estacionamientos": df_raw["Cajones de Estacionamiento"].fillna(0),
        "eje": df_raw["Eje"].astype(str).str.strip(),
        "precio_listado": df_raw["Valor Propiedad"],
        "precio_cerrado": df_raw["Valor Final"],
        "descuento_pct": df_raw["Descuento Total"].fillna(0),
        "fecha_creacion": df_raw["Created Date"],
        "fecha_firma": df_raw["Fecha Firma de Contrato"],
    })

    # Fechas
    for col in ["fecha_creacion", "fecha_firma"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    df["anio_firma"] = df["fecha_firma"].dt.year
    df["mes_firma"] = df["fecha_firma"].dt.month
    df["dias_creacion_a_firma"] = (df["fecha_firma"] - df["fecha_creacion"]).dt.days

    # Filtrar registros válidos
    mask_valid = (
        df["precio_cerrado"].notna()
        & (df["precio_cerrado"] > 0)
        & df["m2_interiores"].notna()
        & (df["m2_interiores"] > 0)
    )
    df = df[mask_valid].copy()

    # Precio por m2 y outliers
    df["precio_m2"] = df["precio_cerrado"] / df["m2_interiores"]
    q1, q99 = df["precio_m2"].quantile([0.01, 0.99])
    df = df[(df["precio_m2"] >= q1) & (df["precio_m2"] <= q99)].copy()

    # ---- Tendencia por desarrollo (CAGR aproximado) ----
    growth_dict = {}
    price_trend = (
        df.groupby(["desarrollo", "anio_firma"])["precio_m2"]
        .median()
        .reset_index()
    )

    for dev, grp in price_trend.groupby("desarrollo"):
        grp = grp.dropna(subset=["anio_firma", "precio_m2"]).sort_values("anio_firma")
        if grp["anio_firma"].nunique() >= 2:
            first = grp["precio_m2"].iloc[0]
            last = grp["precio_m2"].iloc[-1]
            years = grp["anio_firma"].iloc[-1] - grp["anio_firma"].iloc[0]
            if first > 0 and years > 0:
                cagr = (last / first) ** (1 / years) - 1
                growth_dict[dev] = cagr * 100  # en %

    # Crecimiento promedio para fallback
    default_growth = float(np.median(list(growth_dict.values()))) if growth_dict else 5.0

    # Año "hoy" de referencia = último año con datos
    latest_year = int(df["anio_firma"].max())

    return df, growth_dict, default_growth, latest_year


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Entrena un RandomForest sobre el histórico limpio.
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
    cat_features = ["desarrollo"]

    X_num = df[numeric_features].fillna(0)
    X_cat = pd.get_dummies(df[cat_features].astype(str), prefix=cat_features)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["precio_cerrado"]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    feature_columns = X.columns.tolist()
    return model, feature_columns


def build_input_row(
    desarrollo,
    m2_int,
    m2_tot,
    m2_jard,
    m2_terr,
    estac,
    descuento_pct,
    anio_ref,
    mes_ref,
    dias_crea_firma_estimado,
    feature_columns,
):
    """
    Construye la fila de input con el mismo esquema que el training.
    anio_ref / mes_ref se fijan al año más reciente de la base (mercado actual).
    """
    numeric_features = {
        "m2_interiores": m2_int,
        "m2_totales": m2_tot,
        "m2_jardin": m2_jard,
        "m2_terraza": m2_terr,
        "estacionamientos": estac,
        "descuento_pct": descuento_pct,
        "anio_firma": anio_ref,
        "mes_firma": mes_ref,
        "dias_creacion_a_firma": dias_crea_firma_estimado,
    }

    row = pd.DataFrame([numeric_features])

    dummy = pd.get_dummies(
        pd.Series([desarrollo], name="desarrollo").astype(str),
        prefix=["desarrollo"]
    )

    for col in feature_columns:
        if col.startswith("desarrollo_"):
            if col in dummy.columns:
                row[col] = dummy[col].iloc[0]
            else:
                row[col] = 0

    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_columns]
    return row

# =========================
# 2. APP STREAMLIT – ENFOCADA A PRECIOS FUTUROS
# =========================

def main():
    st.set_page_config(
        page_title="AI Pricing Engine – Querétaro",
        layout="wide"
    )

    # Logo opcional
    try:
        logo = Image.open("logo.png")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(logo, width=220)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

    st.title("🏙️ AI Pricing Engine – Querétaro")
    st.caption(
        "Motor de pricing entrenado con ventas reales desde 2013 "
        "para estimar precios actuales y futuros por zona/producto."
    )

    df, growth_dict, default_growth, latest_year = load_and_prepare_data()
    model, feature_columns = train_model(df)

    # ================= SIDEBAR =================
    st.sidebar.header("🎯 Producto objetivo (nuevo proyecto)")

    desarrollo = st.sidebar.selectbox(
        "Zona / desarrollo de referencia",
        sorted(df["desarrollo"].unique())
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
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
        estac = st.number_input(
            "Cajones de estacionamiento",
            min_value=0, max_value=6,
            value=2, step=1
        )
    with c2:
        m2_tot = st.number_input(
            "m² totales (incluye terrazas/jardín)",
            min_value=40.0, max_value=800.0,
            value=130.0, step=5.0
        )
        m2_terr = st.number_input(
            "m² terraza",
            min_value=0.0, max_value=200.0,
            value=8.0, step=2.0
        )
        descuento_pct = st.number_input(
            "Descuento objetivo (%)",
            min_value=0.0, max_value=20.0,
            value=5.0, step=0.5
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Proyección de precios")

    base_growth = growth_dict.get(desarrollo, default_growth)
    supuesto_crec_anual = st.sidebar.slider(
        "Supuesto de crecimiento anual de precios (%)",
        0.0, 15.0,
        float(round(base_growth, 1)) if not np.isnan(base_growth) else 5.0,
        step=0.5,
        help="Puedes usar la sugerencia automática del modelo o ajustarla "
             "considerando inflación + apreciación esperada."
    )

    horizonte_meses = st.sidebar.slider(
        "Horizonte de proyección (meses)",
        0, 36, 12, step=3
    )

    st.sidebar.markdown("---")
    precio_objetivo = st.sidebar.number_input(
        "Tu precio de lista (MXN)",
        min_value=500000.0,
        max_value=40000000.0,
        value=3500000.0,
        step=50000.0
    )

    # ================= PREDICCIÓN =================

    # Referencia de calendario: mercado actual ~ último año de la base
    anio_ref = latest_year
    # Tomamos el mes más frecuente como referencia (estacionalidad media)
    mes_ref = int(df["mes_firma"].dropna().mode().iloc[0])

    # Usamos la mediana de días entre creación y firma como proxy
    dias_medios = int(df["dias_creacion_a_firma"].dropna().median())

    input_row = build_input_row(
        desarrollo=desarrollo,
        m2_int=m2_int,
        m2_tot=m2_tot,
        m2_jard=m2_jard,
        m2_terr=m2_terr,
        estac=estac,
        descuento_pct=descuento_pct,
        anio_ref=anio_ref,
        mes_ref=mes_ref,
        dias_crea_firma_estimado=dias_medios,
        feature_columns=feature_columns,
    )

    precio_hoy = float(model.predict(input_row)[0])
    precio_hoy_min = precio_hoy * 0.95
    precio_hoy_max = precio_hoy * 1.05
    precio_m2_hoy = precio_hoy / m2_int if m2_int > 0 else 0

    # Proyección futura
    factor = (1 + supuesto_crec_anual / 100) ** (horizonte_meses / 12.0)
    precio_futuro = precio_hoy * factor
    precio_futuro_min = precio_hoy_min * factor
    precio_futuro_max = precio_hoy_max * factor
    precio_m2_fut = precio_futuro / m2_int if m2_int > 0 else 0

    delta_abs_hoy = precio_objetivo - precio_hoy
    delta_pct_hoy = (delta_abs_hoy / precio_hoy * 100) if precio_hoy > 0 else 0

    # ================= LAYOUT: TABS =================

    tab_resumen, tab_detalle, tab_hist = st.tabs(
        ["📊 Resumen ejecutivo", "📉 Detalle del escenario", "📈 Historial por zona"]
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
                f"Precio máx. en {horizonte_meses} meses",
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
                "Podría implicar mayor tiempo en mercado o necesidad de descuento."
            )
        else:
            st.info(
                "Tu precio objetivo está por debajo del precio recomendado HOY. "
                "Puede favorecer velocidad de venta o indicar oportunidad de capturar más valor."
            )

        st.markdown(
            f"""
            - El modelo está entrenado con **{len(df):,} operaciones reales**.  
            - La tasa de crecimiento sugerida para **{desarrollo}** es de alrededor de
              **{supuesto_crec_anual:.1f}% anual**, basada en la trayectoria histórica de precio/m².
            """
        )

    # ---- TAB 2: DETALLE ----
    with tab_detalle:
        st.subheader("📉 Detalle del producto y supuestos de proyección")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Configuración física del producto")
            st.write(f"- Zona / desarrollo de referencia: **{desarrollo}**")
            st.write(f"- m² interiores: **{m2_int:.1f} m²**")
            st.write(f"- m² totales: **{m2_tot:.1f} m²**")
            st.write(f"- m² terraza: **{m2_terr:.1f} m²**")
            st.write(f"- m² jardín: **{m2_jard:.1f} m²**")
            st.write(f"- Cajones de estacionamiento: **{estac}**")

        with col_der:
            st.markdown("#### Supuestos comerciales y de mercado")
            st.write(f"- Descuento objetivo vs lista: **{descuento_pct:.1f}%**")
            st.write(f"- Horizonte de proyección: **{horizonte_meses} meses**")
            st.write(f"- Crecimiento anual supuesto: **{supuesto_crec_anual:.1f}%**")
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
        st.subheader("📈 Historial de precios por m² en la zona")

        df_dev = df[df["desarrollo"] == desarrollo].copy()
        if not df_dev.empty:
            st.write(
                f"Historial de ventas para **{desarrollo}** "
                f"({len(df_dev)} operaciones depuradas)."
            )

            st.dataframe(
                df_dev[
                    ["anio_firma", "mes_firma",
                     "m2_interiores", "precio_cerrado", "precio_m2"]
                ].sort_values(["anio_firma", "mes_firma"])
            )

            st.markdown("#### Evolución histórica de precio/m² (mediana anual)")
            pivot = (
                df_dev.groupby("anio_firma")["precio_m2"]
                .median()
                .reset_index()
                .set_index("anio_firma")
            )
            st.line_chart(pivot)
        else:
            st.info("No hay historial suficiente para esta zona en la base.")


if __name__ == "__main__":
    main()
