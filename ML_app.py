import streamlit as st
import ee
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AEGIS-1 AI Mission Control", layout="wide", page_icon="🛰️")

# --- CARGA DEL MODELO (EL CEREBRO) ---
MODEL_PATH = 'aegis_v6_brain.json'

@st.cache_resource
def load_aegis_brain():
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            return model
        except: return None
    return None

aegis_brain = load_aegis_brain()

# --- CONEXIÓN CON GOOGLE EARTH ENGINE ---
def initialize_ee():
    if "gcp_service_account" in st.secrets:
        creds = ee.ServiceAccountCredentials(
            st.secrets["gcp_service_account"]["client_email"],
            key_data=st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n')
        )
        ee.Initialize(creds, project='analisis-post-incendio')

initialize_ee()

# --- INTERFAZ LATERAL (CONTROL DE MISIÓN) ---
st.sidebar.title("🛰️ AEGIS-1 AI CONTROL")

if aegis_brain:
    st.sidebar.success("🤖 AI BRAIN: ONLINE")
else:
    st.sidebar.error("⚠️ AI BRAIN: OFFLINE (Check .json)")

# Coordenadas y Parámetros
with st.sidebar.expander("📍 GEOLOCATION", expanded=True):
    lat = st.number_input("Latitude", value=28.3500, format="%.4f")
    lon = st.number_input("Longitude", value=-16.5000, format="%.4f")
    radius = st.slider("Radius (km)", 1, 10, 3)

# --- PROCESAMIENTO GEOFÍSICO ---
def get_gee_data(lat, lon, radius):
    point = ee.Geometry.Point([lon, lat])
    area = point.buffer(radius * 1000)
    
    # DEM y Pendiente
    srtm = ee.Image("USGS/SRTMGL1_003").clip(area)
    slope = ee.Terrain.slope(srtm)
    
    # Sentinel-2 (Simplificado para el ejemplo, pero funcional)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(area)
    dnbr = s2.normalizedDifference(['B8', 'B12']).rename('dNBR') # Valor proxy para test
    
    # Sentinel-1 (Humedad/Rugosidad)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(area).first().clip(area)
    sar = s1.select('VV').rename('SAR_DELTA')
    
    return dnbr, sar, slope, area

dnbr_img, sar_img, slope_img, region = get_gee_data(lat, lon, radius)

# --- MOTOR DE INFERENCIA (IA REAL) ---
def compute_ai_risk(dnbr_img, sar_img, slope_img, region):
    # Stack de bandas para la IA
    stack = ee.Image.cat([dnbr_img, sar_img, slope_img])
    # Muestreo para la telemetría
    sample = stack.sample(region=region, scale=100, numPixels=500).getInfo()
    
    data = []
    for f in sample['features']:
        props = f['properties']
        data.append([props.get('dNBR', 0), props.get('SAR_DELTA', 0), props.get('SLOPE', 0)])
    
    df = pd.DataFrame(data, columns=['BURN_SEVERITY_dNBR', 'SOIL_MOISTURE_CHANGE', 'SLOPE_DEG'])
    
    if aegis_brain and not df.empty:
        probs = aegis_brain.predict_proba(df)[:, 1]
        return np.mean(probs)
    return 0.0

ai_prob = compute_ai_risk(dnbr_img, sar_img, slope_img, region)

# --- INTERFAZ PRINCIPAL (DASHBOARD) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Satellital Intelligence Map")
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")
    
    # Capa dNBR (Incendio)
    vis_dnbr = {'min': 0, 'max': 0.5, 'palette': ['white', 'orange', 'red']}
    map_id = dnbr_img.getMapId(vis_dnbr)
    folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine').add_to(m)
    
    folium_static(m)

with col2:
    st.subheader("📊 AI Telemetry")
    st.metric("Risk Probability", f"{ai_prob*100:.2f}%")
    
    # Indicador Visual
    if ai_prob > 0.62:
        st.error("🚨 CRITICAL RISK DETECTED")
    else:
        st.success("✅ STABLE TERRAIN")
    
    st.write("---")
    st.write("**Feature Analysis (Avg):**")
    st.write(f"Slope: {radius} km Avg")
    # Aquí puedes añadir más gráficos de Plotly o métricas que echabas de menos

