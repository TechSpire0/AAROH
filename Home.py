# home.py
import streamlit as st
import ee
import geemap.foliumap as geemap
from dotenv import load_dotenv
import os

from llm.agent_tools import get_agent_layer

# Initialize
load_dotenv()
ee.Initialize()

st.set_page_config(layout="wide")
st.title("🌍 AI + Satellite: Smart Spatial Query System")

# --- Step 1: Inputs ---
query = st.text_input("🗣 Ask your spatial question:", "Flood prone areas in Guwahati")

locations_dict = {
    "Guwahati": (91.7362, 26.1445),
    "Hyderabad": (78.4867, 17.3850),
    "Bengaluru": (77.5946, 12.9716),
    "Kolkata": (88.3639, 22.5726),
}

location = st.selectbox("📍 Select a location:", list(locations_dict.keys()))
coords = locations_dict[location]

# Optional buffer slider
buffer_km = st.slider("📏 Select buffer radius (km):", 30, 300, 60)
region = ee.Geometry.Point(coords).buffer(buffer_km * 1000)

# --- Step 2: On Run ---
if st.button("Run Analysis"):
    st.subheader("🧠 LLM Agent Reasoning")
    
    layer, tool_used, debug_log = get_agent_layer(query, coords, buffer_km)

    # --- Step 3: Display Map ---
    st.subheader("🗺 Map Output")
    m = geemap.Map(center=[coords[1], coords[0]], zoom=9)
    vis_params = {"palette": ["#FF0000"], "min": 0, "max": 1}
    if "population" in tool_used.lower():
        vis_params = {"palette": ["white", "red"], "min": 0, "max": 300}
    elif "vegetation" in tool_used.lower():
        vis_params = {"palette": ["#00FF00"], "min": 0, "max": 1}

    m.addLayer(layer, vis_params, tool_used)
    m.to_streamlit(height=600)

    # Safer region check using reduceRegion
    sample = layer.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=layer.geometry(),
        scale=30
    )
    count = sample.getInfo()
    if count and list(count.values())[0] == 0:
        st.warning("⚠ No data detected in this region. Try increasing buffer or relaxing thresholds.")

    st.markdown(f"🔧 Tool selected:** {tool_used}")
    st.code(debug_log, language="text")