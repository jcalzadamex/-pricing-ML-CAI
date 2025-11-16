import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# =========================
# 1. DATA LOADING & MODEL
# =========================

@st.cache_data
def load_data():
    """Load cleaned historical sales data."""
    df = pd.read_csv("data_qro_cleaned.csv")
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train Random Forest model on real sales history."""
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

    df_model = df.copy()

    X_num = df_model[numeric_features].fillna(0)
    X_cat = pd.get_dummies(
        df_model[cat_features].astype(str),
        prefix=cat_features
    )
    X = pd.concat([X_num, X_cat], axis=1)
    y = df_model["precio_cerrado"]

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
    anio_objetivo,
    mes_objetivo,
    dias_crea_firma_estimado,
    feature_columns,
):
    """Build one-row dataframe with same structure as training set."""

    numeric_features = {
        "m2_interiores": m2_int,
        "m2_totales": m2_tot,
        "m2_jardin": m2_jard,
        "m2_terraza": m2_terr,
        "estacionamientos": estac,
        "descuento_pct": descuento_pct,
        "anio_firma": anio_objetivo,
        "mes_firma": mes_objetivo,
        "dias_creacion_a_firma": dias_crea_firma_estimado,
    }

    row = pd.DataFrame([numeric_features])

    # One-hot encoding for desarrollo
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

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_columns]
    return row

# =========================
# 2. STREAMLIT APP
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
        "Modelo de IA entrenado con ventas reales desde 2013 "
        "para recomendar precios de salida y rangos de negociación."
    )

    df = load_data()
    model, feature_columns = train_model(df)

    # ------------- SIDEBAR INPUTS ----------------
    st.sidebar.header("🎯 Parámetros del producto")

    desarrollo = st.sidebar.selectbox(
        "Desarrollo",
        sorted(df["desarrollo"].unique())
    )

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
        estac = st.number_input(
            "Cajones de estacionamiento",
            min_value=0, max_value=5,
            value=2, step=1
        )
    with col2:
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
    st.sidebar.subheader("📅 Supuestos de calendario de venta")

    min_year = int(df["anio_firma"].min())
    max_year = int(df["anio_firma"].max())

    anio_objetivo = st.sidebar.slider(
        "Año estimado de firma",
        min_year, max_year + 1, max_year
    )
    mes_objetivo = st.sidebar.slider(
        "Mes estimado de firma",
        1, 12, 6
    )

    dias_crea_firma_estimado = st.sidebar.slider(
        "Días entre creación y firma",
        int(df["dias_creacion_a_firma"].quantile(0.1)),
        int(df["dias_creacion_a_firma"].quantile(0.9)),
        int(df["dias_creacion_a_firma"].median())
    )

    st.sidebar.markdown("---")
    precio_objetivo = st.sidebar.number_input(
        "Tu precio de lista (MXN)",
        min_value=500000.0,
        max_value=30000000.0,
        value=3500000.0,
        step=50000.0
    )

    # ------------- PREDICCIÓN ----------------

    input_row = build_input_row(
        desarrollo=desarrollo,
        m2_int=m2_int,
        m2_tot=m2_tot,
        m2_jard=m2_jard,
        m2_terr=m2_terr,
        estac=estac,
        descuento_pct=descuento_pct,
        anio_objetivo=anio_objetivo,
        mes_objetivo=mes_objetivo,
        dias_crea_firma_estimado=dias_crea_firma_estimado,
        feature_columns=feature_columns,
    )

    predicted_price = float(model.predict(input_row)[0])
    precio_min = predicted_price * 0.95
    precio_max = predicted_price * 1.05
    precio_m2 = predicted_price / m2_int if m2_int > 0 else 0

    delta_abs = precio_objetivo - predicted_price
    delta_pct = (delta_abs / predicted_price * 100) if predicted_price > 0 else 0

    # ------------- LAYOUT: TABS ----------------
    tab_resumen, tab_detalle, tab_hist = st.tabs(
        ["📊 Resumen ejecutivo", "📉 Detalle de valoración", "📈 Históricos del mercado"]
    )

    # ---- TAB 1: RESUMEN ----
    with tab_resumen:
        st.subheader("📊 Recomendación de precio basada en IA")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Precio recomendado (IA)", f"${predicted_price:,.0f} MXN")
            st.metric("Precio mínimo sugerido", f"${precio_min:,.0f} MXN")
        with c2:
            st.metric("Precio máximo sugerido", f"${precio_max:,.0f} MXN")
            st.metric("Precio objetivo propio", f"${precio_objetivo:,.0f} MXN")
        with c3:
            st.metric("Precio recomendado por m²", f"${precio_m2:,.0f} MXN/m²")
            st.metric("Diferencia vs IA", f"{delta_pct:,.2f} %")

        st.markdown("---")
        if abs(delta_pct) < 3:
            st.success(
                "Tu precio está muy alineado con la señal del modelo. "
                "Es un rango razonable de salida."
            )
        elif delta_pct > 3:
            st.warning(
                "Tu precio está por encima de lo que sugiere el modelo. "
                "Podría implicar mayor tiempo en mercado o necesidad de mayor descuento."
            )
        else:
            st.info(
                "Tu precio está por debajo del rango sugerido. "
                "Podría favorecer velocidad de venta o indicar oportunidad de capturar más valor."
            )

        st.markdown(
            f"""
            - El modelo está entrenado con **{len(df):,} ventas reales** desde 2013.  
            - La recomendación incorpora la historia de precios y descuentos específicos de **{desarrollo}**.
            """
        )

    # ---- TAB 2: DETALLE ----
    with tab_detalle:
        st.subheader("📉 Detalle del producto y contexto")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Configuración física")
            st.write(f"- Desarrollo: **{desarrollo}**")
            st.write(f"- m² interiores: **{m2_int:.1f} m²**")
            st.write(f"- m² totales: **{m2_tot:.1f} m²**")
            st.write(f"- m² terraza: **{m2_terr:.1f} m²**")
            st.write(f"- m² jardín: **{m2_jard:.1f} m²**")
            st.write(f"- Cajones de estacionamiento: **{estac}**")

        with col_der:
            st.markdown("#### Supuestos comerciales")
            st.write(f"- Descuento objetivo: **{descuento_pct:.1f}%**")
            st.write(f"- Año estimado de firma: **{anio_objetivo}**")
            st.write(f"- Mes estimado de firma: **{mes_objetivo}**")
            st.write(
                f"- Días estimados entre creación y firma: "
                f"**{dias_crea_firma_estimado} días**"
            )
            st.write(f"- Precio de lista propuesto: **${precio_objetivo:,.0f} MXN**")
            st.write(
                f"- Diferencia vs precio IA: "
                f"**${delta_abs:,.0f} MXN ({delta_pct:,.2f}%)**"
            )

    # ---- TAB 3: HISTÓRICOS ----
    with tab_hist:
        st.subheader("📈 Históricos por desarrollo")

        df_dev = df[df["desarrollo"] == desarrollo].copy()
        if not df_dev.empty:
            df_dev["precio_m2"] = (
                df_dev["precio_cerrado"] / df_dev["m2_interiores"]
            )
            st.write(
                f"Historial de ventas para **{desarrollo}** "
                f"({len(df_dev)} operaciones)."
            )

            st.dataframe(
                df_dev[
                    ["anio_firma", "mes_firma",
                     "m2_interiores", "precio_cerrado", "precio_m2"]
                ].sort_values(["anio_firma", "mes_firma"])
            )

            st.markdown("#### Evolución de precio por m² (mediana anual)")
            pivot = (
                df_dev.groupby("anio_firma")["precio_m2"]
                .median()
                .reset_index()
                .set_index("anio_firma")
            )
            st.line_chart(pivot)
        else:
            st.info("No hay historial suficiente para este desarrollo en la base.")


if __name__ == "__main__":
    main()
