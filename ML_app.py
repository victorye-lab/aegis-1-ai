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
    page_title="AEGIS-1 | Mission Control", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️"
)

# --- CARGA DEL CEREBRO IA ---
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

# Inicialización de Estado de Sesión
if 'mission_data' not in st.session_state:
    st.session_state.mission_data = None
if 'last_calc_params' not in st.session_state:
    st.session_state.last_calc_params = {}

# BASE DE DATOS DE ESCENARIOS
SCENARIOS = {
    "Custom Location": {
        "coords": [40.416, -3.703],
        "dates": {"pre": (datetime.date(2023,6,1), datetime.date(2023,6,30)), "post": (datetime.date(2023,9,1), datetime.date(2023,9,30)), "event": (datetime.date(2023,10,1), datetime.date(2023,10,15))}
    },
    "S01: Bejís (Castellón, 2022)": {
        "coords": [39.915, -0.700],
        "dates": {"pre": (datetime.date(2022,6,1), datetime.date(2022,6,30)), "post": (datetime.date(2022,8,25), datetime.date(2022,8,30)), "event": (datetime.date(2022,9,1), datetime.date(2022,9,15))}
    },
    "S02: Vall d'Ebo (Alicante, 2022)": {
        "coords": [38.805, -0.160],
        "dates": {"pre": (datetime.date(2022,5,1), datetime.date(2022,6,30)), "post": (datetime.date(2022,8,15), datetime.date(2022,8,30)), "event": (datetime.date(2022,8,25), datetime.date(2022,9,5))}
    },
    "S03: Tenerife (Canarias, 2023)": {
        "coords": [28.35, -16.50],
        "dates": {"pre": (datetime.date(2023,5,1), datetime.date(2023,6,30)), "post": (datetime.date(2023,8,20), datetime.date(2023,8,30)), "event": (datetime.date(2023,9,1), datetime.date(2023,9,15))}
    }
}

def initialize_geospatial_engine():
    project_id = 'analisis-post-incendio'
    if "gcp_service_account" in st.secrets:
        try:
            raw_key = st.secrets["gcp_service_account"]["private_key"]
            parsed_key = raw_key.replace('\\n', '\n') if '\\n' in raw_key else raw_key
            credentials = ee.ServiceAccountCredentials(st.secrets["gcp_service_account"]["client_email"], key_data=parsed_key)
            ee.Initialize(credentials, project=project_id)
        except Exception as e:
            st.error(f"Fallo Crítico: {e}"); st.stop()
    else:
        ee.Initialize(project=project_id)

initialize_geospatial_engine()

# ---------------------------------------------------------
# 2. UI THEME ENGINE
# ---------------------------------------------------------
def apply_professional_theme():
    bg, sidebar_bg, text, card_bg, border, accent = "#0e1117", "#262730", "#fafafa", "#1f2129", "#444", "#00FFFF"
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Roboto+Mono:wght@400;700&display=swap');
        .stApp {{ background-color: {bg}; color: {text}; font-family: 'Inter', sans-serif; }}
        section[data-testid="stSidebar"] > div {{ background-color: {sidebar_bg}; border-right: 1px solid {border}; }}
        h1, h2, h3 {{ font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 1px; color: {text} !important; }}
        div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono', monospace; font-size: 1.5rem !important; color: {text} !important; }}
        div[data-testid="stMetric"] {{ background-color: {card_bg}; border: 1px solid {border}; padding: 10px; border-radius: 4px; }}
        .legend-container {{ background-color: {card_bg}; padding: 10px; border: 1px solid {border}; border-radius: 4px; margin-top: 10px; }}
        .live-indicator {{ animation: blink 2s infinite; color: {accent}; font-weight: bold; }}
        @keyframes blink {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} 100% {{ opacity: 1; }} }}
    </style>
    """, unsafe_allow_html=True)
    return text

def create_categorical_legend(text_color):
    return f"""
    <div class="legend-container">
        <div style="font-family: 'Roboto Mono', monospace; font-size: 0.7em; font-weight: bold; color: {text_color}; margin-bottom: 8px; border-bottom: 1px solid #555; padding-bottom: 4px; display: flex; justify-content: space-between;">
            <span>RISK CLASSIFICATION (v6.4.2-AI)</span>
            <span class="live-indicator">● AI ACTIVE</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr; gap: 5px; font-family: 'Inter', sans-serif; font-size: 0.75em; color: {text_color};">
             <div style="display: flex; align-items: center;"><span style="width: 12px; height: 12px; background: #FF00FF; margin-right: 8px; border: 1px solid white;"></span>CRITICAL (>0.8) - Debris Flow</div>
             <div style="display: flex; align-items: center;"><span style="width: 12px; height: 12px; background: #FF0000; margin-right: 8px;"></span>HIGH (0.6 - 0.8) - Danger</div>
             <div style="display: flex; align-items: center;"><span style="width: 12px; height: 12px; background: #FFFF00; margin-right: 8px;"></span>MODERATE (0.4 - 0.6) - Caution</div>
             <div style="display: flex; align-items: center;"><span style="width: 12px; height: 12px; background: #00FFFF; margin-right: 8px; border: 1px solid #555;"></span>DETECTED MOISTURE (Solid Radar)</div>
        </div>
    </div>
    """

# ---------------------------------------------------------
# 3. GEOSPATIAL CORE (v5.5.6 INTEGRAL)
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_risk_analysis(lat, lon, date_pre, date_post, date_storm, min_display_threshold, buffer_km):
    try:
        point = ee.Geometry.Point([lon, lat])
        analysis_buffer = point.buffer(buffer_km * 1000)
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).clip(analysis_buffer)
        wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").clip(analysis_buffer)
        water_mask = jrc.lt(10)
        is_urban = wc.eq(50); is_agri = wc.eq(40)
        natural_land_mask = wc.neq(80).And(wc.neq(50)).And(wc.neq(40)).And(water_mask)

        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(['B4', 'B8', 'B11', 'B12', 'QA60'])
        def mask_clouds_fast(img):
            qa = img.select('QA60')
            return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)))
        
        s2_pre = s2.filterBounds(analysis_buffer).filterDate(date_pre[0], date_pre[1]).map(mask_clouds_fast).median().clip(analysis_buffer)
        s2_post = s2.filterBounds(analysis_buffer).filterDate(date_post[0], date_post[1]).map(mask_clouds_fast).median().clip(analysis_buffer)

        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12'])).rename('dNBR')
        ndvi_post = s2_post.normalizedDifference(['B8', 'B4']); ndvi_pre = s2_pre.normalizedDifference(['B8', 'B4'])
        dndvi = ndvi_pre.subtract(ndvi_post)
        
        raw_fire_mask = dnbr.gt(0.15).And(ndvi_post.lt(0.3)).And(dndvi.gt(0.1)).And(natural_land_mask)
        legacy_fire_mask = raw_fire_mask.updateMask(raw_fire_mask.connectedPixelCount(100, True).gte(50))

        s1 = ee.ImageCollection('COPERNICUS/S1_GRD').select(['VV']).filterBounds(analysis_buffer).filter(ee.Filter.eq('instrumentMode', 'IW'))
        s1_dry = s1.filterDate(date_post[0], date_post[1]).mosaic().clip(analysis_buffer)
        s1_wet = s1.filterDate(date_storm[0], date_storm[1]).mosaic().clip(analysis_buffer)
        sar_delta = s1_wet.subtract(s1_dry).rename('SAR_DELTA')
        slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003').clip(analysis_buffer)).rename('SLOPE')
        
        integrated_mask = legacy_fire_mask.Or(sar_delta.gt(2.5).And(is_urban.Or(is_agri)).And(legacy_fire_mask.focal_max(500)))
        risk_index = dnbr.unitScale(0.15, 0.7).multiply(0.42).add(sar_delta.unitScale(0.5, 2.5).multiply(0.33)).add(slope.unitScale(5, 35).multiply(0.25)).rename('RISK_SCORE')
        
       # --- EXPORT STACK v6.4.4 (OPTIMIZADO PARA SERIALIZACIÓN) ---
        # Usamos .float() para reducir el peso del objeto y asegurar compatibilidad
        export_stack = ee.Image.cat([
            dnbr.float().rename('dNBR'),
            sar_delta.float().rename('SAR_DELTA'),
            slope.float().rename('SLOPE')
        ]).unmask(0) # Crucial: elimina valores null que rompen el encoder
        return {
            "risk_layer": risk_index.updateMask(integrated_mask).updateMask(risk_index.gte(min_display_threshold)),
            "dnbr_layer": dnbr.updateMask(legacy_fire_mask), 
            "sar_layer": sar_delta.gt(0.8).updateMask(integrated_mask),
            "raw_risk": risk_index.updateMask(integrated_mask),
            "export_stack": export_stack,
            "analysis_buffer": analysis_buffer,
            "is_urban": is_urban,
            "integrated_mask": integrated_mask,
            "geom": point, "success": True
        }
    except Exception as e: return {"error": str(e), "success": False}

# --- MAIN ---
def main():
    text_color = apply_professional_theme()
    with st.sidebar:
        st.markdown("### 📡 SYSTEM CONFIGURATION")
        if aegis_brain: st.success("🤖 AI BRAIN LOADED")
        selected_mission = st.selectbox("MISSION PROFILE", list(SCENARIOS.keys()))
        mission_params = SCENARIOS[selected_mission]
        st.markdown("---")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", value=float(mission_params["coords"][0]), format="%.4f")
        lon = c2.number_input("Lon", value=float(mission_params["coords"][1]), format="%.4f")
        buffer_km = st.slider("Analysis Radius (km)", 2, 50, 3)
        st.markdown("---")
        with st.expander("⏳ TIME WINDOWS"):
            d_pre = st.date_input("Pre-Fire", mission_params["dates"]['pre'])
            d_post = st.date_input("Post-Fire", mission_params["dates"]['post'])
            d_storm = st.date_input("Storm", mission_params["dates"]['event'])
        st.markdown("---")
        min_risk_threshold = st.slider("Risk Filter", 0.0, 0.8, 0.35)
        view_mode = st.selectbox("Map Type", ["Satellite (Realism)", "Dark (Technical)", "Light (Day Ops)"])
        visible_layers = st.multiselect("Active Layers", ['Risk Model (WLC)', 'Soil Moisture (Radar)', 'Burn Scar (Orange)'], default=['Burn Scar (Orange)'])

    st.title("AEGIS-1 // RISK INTELLIGENCE PLATFORM")
    st.markdown("##### POST-WILDFIRE GEOMORPHOLOGICAL SURVEILLANCE")
    
    col_map, col_data = st.columns([3, 1])
    
    # --- FORMATEO ESTRICTO DE FECHAS (v6.4.7) ---
    def format_date_for_gee(d):
        if isinstance(d, tuple) and len(d) == 2:
            return (str(d[0]), str(d[1]))
        elif isinstance(d, tuple) and len(d) == 1:
            return (str(d[0]), str(d[0]))
        else:
            return (str(d), str(d))

    dates_fmt = {
        'pre': format_date_for_gee(d_pre),
        'post': format_date_for_gee(d_post),
        'storm': format_date_for_gee(d_storm)
    }
    
    current_params = {'lat': lat, 'lon': lon, 'dates': str(dates_fmt), 'buffer': buffer_km}
    
    if st.session_state.mission_data is None or st.session_state.last_calc_params != current_params:
        with st.status("🛰️ UPDATING MISSION DATA..."):
            # Usamos dates_fmt en lugar de pasar d_pre crudo
            data = generate_risk_analysis(
                lat, lon, 
                dates_fmt['pre'], 
                dates_fmt['post'], 
                dates_fmt['storm'], 
                min_risk_threshold, 
                buffer_km
            )
            st.session_state.mission_data = data
            st.session_state.last_calc_params = current_params

    data = st.session_state.mission_data

    with col_map:
        if data.get("success"):
            if "Satellite" in view_mode: 
                m = folium.Map([lat, lon], zoom_start=12, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')
            elif "Light" in view_mode: 
                m = folium.Map([lat, lon], zoom_start=12, tiles="OpenStreetMap")
            else: 
                m = folium.Map([lat, lon], zoom_start=12, tiles="CartoDB dark_matter")

            def add_lyr(img, name, p):
                try: 
                    folium.raster_layers.TileLayer(tiles=ee.Image(img).getMapId(p)['tile_fetcher'].url_format, attr='GEE', name=name, overlay=True).add_to(m)
                except: pass

            if 'Burn Scar (Orange)' in visible_layers: add_lyr(data['dnbr_layer'], 'Burn Scar', {'min':0.1, 'max':0.7, 'palette':['FF4500', '8B0000']})
            if 'Risk Model (WLC)' in visible_layers: add_lyr(data['risk_layer'], 'Risk Model', {'min': 0.35, 'max': 0.8, 'palette': ['FFFF00', 'FF0000', 'FF00FF']})
            if 'Soil Moisture (Radar)' in visible_layers: add_lyr(data['sar_layer'], 'Moisture', {'palette':['00FFFF']})
            
            folium.LayerControl().add_to(m)
            st.components.v1.html(m._repr_html_(), height=850)

    with col_data:
        if data.get("success"):
            st.markdown("### MISSION TELEMETRY")
            if aegis_brain:
                with st.spinner("🤖 AI INFERENCE..."):
                    try:
                        # Usamos .bounds() para enviar un rectángulo simple, no el círculo complejo
                        roi = data['analysis_buffer'].bounds()
                        
                        # Muestreo optimizado
                        samples = data['export_stack'].sample(
                            region=roi, 
                            scale=200, 
                            numPixels=300,
                            geometries=False # Esto reduce el peso del JSON masivamente
                        ).getInfo()
                        
                        if 'features' in samples and len(samples['features']) > 0:
                            # Extracción de propiedades
                            feat_list = [[
                                f['properties'].get('dNBR', 0), 
                                f['properties'].get('SAR_DELTA', 0), 
                                f['properties'].get('SLOPE', 0)
                            ] for f in samples['features']]
                            
                            input_df = pd.DataFrame(feat_list, columns=['BURN_SEVERITY_dNBR', 'SOIL_MOISTURE_CHANGE', 'SLOPE_DEG'])
                            
                            # Ejecución del cerebro XGBoost
                            ai_prob = np.mean(aegis_brain.predict_proba(input_df)[:, 1])
                            
                            st.metric("AI PREDICTION (XGBOOST)", f"{ai_prob*100:.1f}%", delta="Neural Scan")
                        else:
                            st.warning("⚠️ IA: Zona sin datos válidos.")
                    except Exception as e:
                        st.error("⚠️ IA: Error de Serialización")
                        # Imprimimos el error simplificado para no ensuciar la UI
                        st.caption(f"Log: {str(e)[:100]}")

            try:
                stats = data['raw_risk'].reduceRegion(ee.Reducer.mean(), data['geom'].buffer(1000), 250).getInfo()
                val = stats.get('RISK_SCORE', 0) if stats else 0
                st.metric("VIEWPORT RISK (v5.5)", f"{(val if val else 0)*100:.1f}%", delta="Static WLC")
            except: st.metric("VIEWPORT RISK (v5.5)", "N/A")

            st.markdown(create_categorical_legend(text_color), unsafe_allow_html=True)
            if st.button("GENERATE DATASET (CSV)"):
                try:
                    df = pd.DataFrame([[f['geometry']['coordinates'][1], f['geometry']['coordinates'][0], f['properties'].get('dNBR', 0)] for f in data['export_stack'].sample(region=data['analysis_buffer'], scale=100, numPixels=1000).getInfo()['features']], columns=['LAT','LON','dNBR'])
                    st.download_button("DOWNLOAD CSV", df.to_csv(index=False), "aegis_extract.csv", "text/csv")
                except: st.error("Export Error")

if __name__ == "__main__":
    main()



