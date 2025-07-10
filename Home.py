import streamlit as st
import ee
import geemap.foliumap as geemap
from dotenv import load_dotenv  
import os

from llm.agent_tools import get_agent_layer

# --- Initialization ---
load_dotenv()  # Load environment variables from .env file
ee.Initialize()  # Initialize Earth Engine

# --- Streamlit UI setup ---
st.set_page_config(layout="wide")
st.title("🌍 AAROH (AI-Assisted Reasoning for Orchestrated Geospatial Handling)")
st.markdown("##### This is a demo of the application")

# --- Step 1: User Input ---
query = st.text_input("🗣 Ask your spatial question:", "Flood prone areas in Guwahati")

# Predefined city coordinates for quick matching
locations_dict = {
    "Hyderabad": (78.4867, 17.3850),
    "Bengaluru": (77.5946, 12.9716),
    "Delhi": (77.1025, 28.7041),
    "Mumbai": (72.8777, 19.0760),
    "Chennai": (80.2707, 13.0827),
    "Kolkata": (88.3639, 22.5726),
    "Guwahati": (91.7362, 26.1445),
}

# --- Step 2: Match Place from Query ---
matched_city = None
for city in locations_dict:
    # Check if city is mentioned in the query
    if city.lower() in query.lower():
        matched_city = city
        coords = locations_dict[city]
        break

# If no valid city is found, show error and stop
if not matched_city:
    st.error("❌ Could not detect a valid city in your query. Please include one of the following: Guwahati, Hyderabad, Delhi, Mumbai, Chennai, Bengaluru, or Kolkata.")
    st.stop()

# --- Step 3: Buffer selection and region geometry ---
buffer_km = st.slider("📏 Select buffer radius (km):", 30, 160, 100)
region = ee.Geometry.Point(coords).buffer(buffer_km * 1000)

# --- Step 4: Run Analysis on button click ---
if st.button("Run Analysis"):
    # Call the LLM agent to get analysis layer and reasoning
    layer, tool_used, response, reasoning_steps = get_agent_layer(query, coords, buffer_km, matched_city)

    st.markdown("### 🧠 Step-by-step LLM Agent reasoning")
    for i, step in enumerate(reasoning_steps, 1):
        st.markdown(f"**{i}. {step}**")

    # --- Step 5: Display Map ---
    st.subheader("🗺 Map Output")
    m = geemap.Map(center=[coords[1], coords[0]], zoom=9)

    # Set visualization styles based on tool used
    if "vegetation" in tool_used.lower() or "ndvi" in tool_used.lower():
        vis_params = {"palette": ["#00FF00"], "min": 0, "max": 1}
    elif "solar" in tool_used.lower():
        vis_params = {"palette": ["#fff5b1", "#f18f01", "#a70000"], "min": 100, "max": 300}
    elif "land cover" in tool_used.lower():
        vis_params = {
            "min": 10, "max": 100,
            "palette": ["#006400", "#00FF00", "#7FFF00", "#FFFF00", "#FF7F50", "#D2691E"]
        }
    elif "water" in tool_used.lower():
        vis_params = {"palette": ["#0000FF"]}
    else:  # fallback/flood
        vis_params = {"palette": ["#FF0000"], "min": 0, "max": 1}

    m.addLayer(layer, vis_params, tool_used)
    m.to_streamlit(height=600)

    # Safer region check using reduceRegion to count pixels in the region
    sample = layer.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=layer.geometry(),
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    )
    count = sample.getInfo()
    if count and list(count.values())[0] == 0:
        st.warning("⚠️ No data detected in this region. Try increasing buffer or relaxing thresholds.")

