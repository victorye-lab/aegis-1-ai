import streamlit as st
import ee
import folium
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ---------------------------------------------------------
# 1. SYSTEM CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AEGIS-1 | Smart Mission Control", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️"
)

# Constantes de Inferencia
MODEL_PATH = 'aegis_v6_brain.json'
THRESHOLD = 0.62  # Validado en v6.1.0-beta

# Inicialización de Estado de Sesión
if 'mission_data' not in st.session_state:
    st.session_state.mission_data = None
if 'last_calc_params' not in st.session_state:
    st.session_state.last_calc_params = {}

# Carga del Cerebro IA
@st.cache_resource
def load_aegis_brain():
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model
    return None

aegis_brain = load_aegis_brain()

# BASE DE DATOS DE ESCENARIOS (S01-S03)
SCENARIOS = {
    "Custom Location": {
        "coords": [40.416, -3.703],
        "dates": {"pre": (datetime.date(2023,6,1), datetime.date(2023,6,30)), "post": (datetime.date(2023,9,1), datetime.date(2023,9,30)), "event": (datetime.date(2023,10,1), datetime.date(2023,10,15))}
    },
    "S03: Tenerife (Canarias, 2023)": {
        "coords": [28.35, -16.50],
        "dates": {"pre": (datetime.date(2023,5,1), datetime.date(2023,6,30)), "post": (datetime.date(2023,8,20), datetime.date(2023,8,30)), "event": (datetime.date(2023,9,1), datetime.date(2023,9,15))}
    }
}

def initialize_geospatial_engine():
    project_id = 'analisis-post-incendio'
    if "gcp_service_account" in st.secrets:
        raw_key = st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n')
        credentials = ee.ServiceAccountCredentials(st.secrets["gcp_service_account"]["client_email"], key_data=raw_key)
        ee.Initialize(credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)

initialize_geospatial_engine()

# ---------------------------------------------------------
# 2. IA INFERENCE CORE (v6.2.0)
# ---------------------------------------------------------
def run_ia_inference(export_stack, analysis_buffer):
    """
    Convierte el mapa de píxeles en una predicción de la IA
    """
    if aegis_brain is None:
        return None

    # Muestreo denso para generar el mapa predictivo
    samples = export_stack.sample(region=analysis_buffer, scale=100, numPixels=3000, geometries=True).getInfo()
    
    features = []
    for f in samples['features']:
        p = f['properties']
        features.append([p.get('dNBR', 0), p.get('SAR_DELTA', 0), p.get('SLOPE', 0)])
    
    if not features: return None
    
    # Inferencia
    X = pd.DataFrame(features, columns=['BURN_SEVERITY_dNBR', 'SOIL_MOISTURE_CHANGE', 'SLOPE_DEG'])
    probs = aegis_brain.predict_proba(X)[:, 1]
    
    # El valor medio de probabilidad de la zona
    return np.mean(probs)

# ---------------------------------------------------------
# 3. GEOSPATIAL ANALYSIS
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_risk_analysis(lat, lon, date_pre, date_post, date_storm, buffer_km):
    try:
        point = ee.Geometry.Point([lon, lat])
        analysis_buffer = point.buffer(buffer_km * 1000)
        
        # Datasets
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).clip(analysis_buffer)
        wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").clip(analysis_buffer)
        
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(analysis_buffer)
        s2_pre = s2.filterDate(date_pre[0], date_pre[1]).median().clip(analysis_buffer)
        s2_post = s2.filterDate(date_post[0], date_post[1]).median().clip(analysis_buffer)

        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12'])).rename('dNBR')
        
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(analysis_buffer).filter(ee.Filter.eq('instrumentMode', 'IW'))
        s1_dry = s1.filterDate(date_post[0], date_post[1]).mosaic().clip(analysis_buffer)
        s1_wet = s1.filterDate(date_storm[0], date_storm[1]).mosaic().clip(analysis_buffer)
        sar_delta = s1_wet.select('VV').subtract(s1_dry.select('VV')).rename('SAR_DELTA')
        
        slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003').clip(analysis_buffer)).rename('SLOPE')

        # Export Stack para IA
        export_stack = ee.Image.cat([dnbr, sar_delta, slope])
        
        # Ejecutar Inferencia para telemetría
        ai_risk_val = run_ia_inference(export_stack, analysis_buffer)

        # Capas visuales (Usamos dNBR y SAR para visualización base)
        fire_mask = dnbr.gt(0.15)
        risk_vis = dnbr.unitScale(0.15, 0.7).updateMask(fire_mask).updateMask(slope.gt(15))

        return {
            "risk_layer": risk_vis,
            "dnbr_layer": dnbr.updateMask(fire_mask),
            "ai_score": ai_risk_val,
            "success": True,
            "export_stack": export_stack,
            "analysis_buffer": analysis_buffer,
            "geom": point
        }
    except Exception as e:
        return {"error": str(e), "success": False}

# ---------------------------------------------------------
# 4. MAIN UI
# ---------------------------------------------------------
def main():
    st.sidebar.title("📡 AEGIS-1 v6.2.0")
    if aegis_brain:
        st.sidebar.success("BRAIN LOADED: XGBoost Active")
    else:
        st.sidebar.error("BRAIN NOT FOUND: Using Static Fallback")

    selected_mission = st.sidebar.selectbox("MISSION PROFILE", list(SCENARIOS.keys()))
    m_params = SCENARIOS[selected_mission]
    
    lat = st.sidebar.number_input("Lat", value=m_params["coords"][0], format="%.4f")
    lon = st.sidebar.number_input("Lon", value=m_params["coords"][1], format="%.4f")
    buffer_km = st.sidebar.slider("Radius (km)", 2, 20, 3)

    st.title("AEGIS-1 // SMART SURVEILLANCE")
    
    # Lógica de cálculo
    data = generate_risk_analysis(lat, lon, m_params["dates"]["pre"], m_params["dates"]["post"], m_params["dates"]["event"], buffer_km)

    col1, col2 = st.columns([3, 1])

    with col1:
        m = folium.Map([lat, lon], zoom_start=13, tiles="CartoDB dark_matter")
        if data["success"]:
            # Pintar dNBR (Escenario de Incendio)
            map_id = data["dnbr_layer"].getMapId({'min': 0.1, 'max': 0.7, 'palette': ['orange', 'red']})
            folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='GEE').add_to(m)
        st.components.v1.html(m._repr_html_(), height=600)

    with col2:
        st.subheader("IA TELEMETRY")
        if data["success"]:
            score = data["ai_score"] if data["ai_score"] else 0
            st.metric("AI COLLAPSE PROB", f"{score*100:.1f}%")
            
            status = "STABLE"
            color = "green"
            if score > THRESHOLD:
                status = "CRITICAL"
                color = "red"
            
            st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)
            st.caption(f"Decision Threshold: {THRESHOLD}")

if __name__ == "__main__":
    main()