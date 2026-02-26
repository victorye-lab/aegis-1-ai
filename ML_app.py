import streamlit as st
import ee
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AEGIS-1 | Smart Mission Control", layout="wide", page_icon="🛰️")

# --- CARGA DEL CEREBRO IA ---
MODEL_PATH = 'aegis_v6_brain.json'
@st.cache_resource
def load_aegis_brain():
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error cargando el cerebro IA: {e}")
            return None
    return None

aegis_brain = load_aegis_brain()

# --- CONEXIÓN GOOGLE EARTH ENGINE ---
def initialize_ee():
    if "gcp_service_account" in st.secrets:
        raw_key = st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n')
        creds = ee.ServiceAccountCredentials(st.secrets["gcp_service_account"]["client_email"], key_data=raw_key)
        ee.Initialize(creds, project='analisis-post-incendio')

initialize_ee()

# --- INTERFAZ LATERAL COMPLETA (SIN RECORTES) ---
st.sidebar.title("🛰️ AEGIS-1 CONTROL PANEL")
st.sidebar.markdown("---")

# Estado de la IA
if aegis_brain:
    st.sidebar.success("🤖 AI ENGINE: ONLINE")
else:
    st.sidebar.warning("⚠️ AI ENGINE: OFFLINE (Usando lógica estática)")

# Parámetros de Misión
st.sidebar.subheader("📍 Mission Parameters")
lat = st.sidebar.number_input("Latitude", value=28.3500, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-16.5000, format="%.4f")
radius = st.sidebar.slider("Analysis Radius (km)", 1, 10, 3)

st.sidebar.subheader("📅 Temporal Analysis")
date_pre = st.sidebar.date_input("Pre-Fire Date", datetime(2023, 6, 1))
date_post = st.sidebar.date_input("Post-Fire Date", datetime(2023, 8, 25))
date_storm = st.sidebar.date_input("Storm Event", datetime(2023, 9, 5))

# --- PROCESAMIENTO GEOFÍSICO ---
def get_analysis(lat, lon, r, d_pre, d_post, d_storm):
    area = ee.Geometry.Point([lon, lat]).buffer(r * 1000)
    
    # DEM y Pendiente
    srtm = ee.Image("USGS/SRTMGL1_003").clip(area)
    slope = ee.Terrain.slope(srtm)
    
    # Incendio (dNBR)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)
    pre_fire = s2.filterDate(str(d_pre - timedelta(days=30)), str(d_pre)).median().clip(area)
    post_fire = s2.filterDate(str(d_post), str(d_post + timedelta(days=30))).median().clip(area)
    dnbr = pre_fire.normalizedDifference(['B8', 'B12']).subtract(post_fire.normalizedDifference(['B8', 'B12'])).rename('dNBR')
    
    # Humedad (SAR Delta)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(area).filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_dry = s1.filterDate(str(d_post), str(d_post + timedelta(days=10))).median().clip(area)
    s1_wet = s1.filterDate(str(d_storm), str(d_storm + timedelta(days=5))).median().clip(area)
    sar_delta = s1_wet.select('VV').subtract(s1_dry.select('VV')).rename('SAR_DELTA')
    
    return dnbr, sar_delta, slope, area

dnbr_img, sar_img, slope_img, region = get_analysis(lat, lon, radius, date_pre, date_post, date_storm)

# --- INFERENCIA IA ---
def run_ai_prediction(dnbr, sar, slope, region):
    stack = ee.Image.cat([dnbr, sar, slope])
    sample = stack.sample(region=region, scale=100, numPixels=800).getInfo()
    
    features = []
    for f in sample['features']:
        p = f['properties']
        features.append([p.get('dNBR', 0), p.get('SAR_DELTA', 0), p.get('SLOPE', 0)])
    
    df = pd.DataFrame(features, columns=['BURN_SEVERITY_dNBR', 'SOIL_MOISTURE_CHANGE', 'SLOPE_DEG'])
    
    if aegis_brain and not df.empty:
        probs = aegis_brain.predict_proba(df)[:, 1]
        return np.mean(probs)
    return 0.0

ai_risk = run_ai_prediction(dnbr_img, sar_img, slope_img, region)

# --- DASHBOARD PRINCIPAL ---
st.title("🛰️ AEGIS-1: Smart Post-Fire Monitoring")
st.markdown(f"**Current Status:** {'🔴 CRITICAL RISK' if ai_risk > 0.62 else '🟢 STABLE TERRAIN'}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌍 Geospatial Intelligence Map")
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")
    
    # Capas Visuales
    vis_dnbr = {'min': 0.1, 'max': 0.6, 'palette': ['green', 'yellow', 'orange', 'red']}
    map_id = dnbr_img.getMapId(vis_dnbr)
    folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine dNBR', name="Burn Severity").add_to(m)
    
    folium.LayerControl().add_to(m)
    folium_static(m)

with col2:
    st.subheader("📊 AI Analytics & Telemetry")
    st.metric("AI Collapse Probability", f"{ai_risk*100:.2f}%")
    st.progress(min(float(ai_risk), 1.0))
    
    st.write("---")
    st.write("**Environmental Indicators (Mean)**")
    # Cálculos rápidos para métricas
    mean_slope = slope_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=100).getInfo().get('SLOPE', 0)
    st.metric("Avg. Slope", f"{mean_slope:.2f}°")
    
    st.write("---")
    st.info("💡 **AI Insight:** El modelo XGBoost está analizando la correlación entre la severidad del incendio y el cambio de humedad en el suelo para predecir colapsos.")
