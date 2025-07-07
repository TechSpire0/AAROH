from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os
import ee
import streamlit as st
from rag.retriever import retrieve_similar_example
from gee.flood import (
    get_flood_mask, get_ndvi_mask, get_s1_water_mask, get_peak_ndvi,
    get_solar_irradiance, get_land_cover
)

load_dotenv()

# ✅ Set up LLM (Mistral via Together API)
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.0,
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1"
)

# ✅ Store tool used
last_tool_used = {"name": None}

def tool_wrapper(func, region, name):
    def wrapped(_):
        last_tool_used["name"] = name
        return func(region)
    return wrapped

# ✅ Define GEE tools for the agent
def get_tools(region):
    return [
        Tool.from_function(
            name="get_flood_mask",
            func=tool_wrapper(get_flood_mask, region, "get_flood_mask"),
            description="Use this to detect flood-prone terrain using elevation < 200m and slope < 5 degrees."
        ),
        Tool.from_function(
            name="get_ndvi_mask",
            func=tool_wrapper(get_ndvi_mask, region, "get_ndvi_mask"),
            description="Use this to find areas with low vegetation using NDVI < 0.2 from Sentinel-2."
        ),
        Tool.from_function(
            name="get_s1_water_mask",
            func=tool_wrapper(get_s1_water_mask, region, "get_s1_water_mask"),
            description="Use this to detect flooded areas using Sentinel-1 radar data."
        ),
        Tool.from_function(
            name="get_peak_ndvi",
            func=tool_wrapper(get_peak_ndvi, region, "get_peak_ndvi"),
            description="Use this to get peak crop NDVI over a year from Sentinel-2."
        ),
        Tool.from_function(
            name="get_solar_irradiance",
            func=tool_wrapper(get_solar_irradiance, region, "get_solar_irradiance"),
            description="Use this to get solar irradiance (solar energy potential) from NASA POWER."
        ),
        Tool.from_function(
            name="get_land_cover",
            func=tool_wrapper(get_land_cover, region, "get_land_cover"),
            description="Use this to view land cover classification from ESA WorldCover dataset."
        )
    ]

# ✅ Run the agent
def get_agent_layer(query, coords, buffer_km=30, matched_city="Unknown"):
    region = ee.Geometry.Point(coords).buffer(buffer_km * 1000)
    tools = get_tools(region)

    example = retrieve_similar_example(query)
    example_block = ""
    if example is not None:
        example_block = (
            f"Example:\n"
            f"Query: {example['query']}\n"
            f"Tool: {example['tool_name']}\n"
            f"Reasoning: {example['reasoning']}\n"
            f"Action Input: {example['action_input']}\n\n"
    )

    system_message = SystemMessage(
        content=(
            "You are a spatial analysis assistant that uses Earth Engine tools to answer geographic queries.\n"
            "Use ONLY the provided tools to answer the query.\n"
            "Only use one tool that best answers the query.\n\n"
            + example_block +
            f"Now answer this new query:\nQuery: {query}"
    )
)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        max_iterations=4,
        max_execution_time=60,
        system_message=system_message
    )

    try:
        response = agent.run(query)
        tool_name = last_tool_used["name"]
    except OutputParserException as e:
        response = str(e)
        tool_name = "get_flood_mask"
        layer = get_flood_mask(region)
        tool_explanation = "🛠️ The tool **get_flood_mask** was used as a fallback. It estimates flood-prone terrain using low elevation and slope."
        reasoning_steps = [
            f"⚠️ The AI had trouble understanding the query. As a fallback, flood-prone analysis was performed.",
            f"📍 The analysis was centered on coordinates: {coords}.",
            f"📏 A region with **{buffer_km} km** buffer was selected.",
            f"{tool_explanation}",
            f"⚙️ The method involved filtering terrain below 200m elevation and slope under 10°.",
            f"🗺️ A red overlay highlights zones likely to flood.",
        ]
        return layer, "Flood-Prone Terrain (Fallback)", response, reasoning_steps

    # ✅ Match tool name to function for output
    tool_map = {
        "get_flood_mask": (get_flood_mask(region), "Flood-Prone Terrain"),
        "get_ndvi_mask": (get_ndvi_mask(region), "Low Vegetation Zones"),
        "get_s1_water_mask": (get_s1_water_mask(region), "Radar-based Water Detection"),
        "get_peak_ndvi": (get_peak_ndvi(region), "Peak NDVI (Crop Growth)"),
        "get_solar_irradiance": (get_solar_irradiance(region), "Solar Irradiance"),
        "get_land_cover": (get_land_cover(region), "Land Cover Classification"),
    }

    layer, tool_label = tool_map.get(tool_name, (get_flood_mask(region), "Flood-Prone Terrain (Fallback)"))

    # ✅ Generate reasoning steps
    tool_explanation_map = {
    "get_flood_mask": (
        "🛠️ The tool **get_flood_mask** was used. It analyzes elevation and terrain slope to identify low-lying, flat areas below 200 meters elevation and less than 10° slope, which are prone to flooding."
    ),
    "get_ndvi_mask": (
        "🛠️ The tool **get_ndvi_mask** was used. It calculates the NDVI (Normalized Difference Vegetation Index) using Sentinel-2 satellite images to highlight areas with sparse or unhealthy vegetation."
    ),
    "get_s1_water_mask": (
        "🛠️ The tool **get_s1_water_mask** was used. It processes Sentinel-1 radar data to detect water surfaces, including flooded regions, by identifying low radar backscatter values."
    ),
    "get_peak_ndvi": (
        "🛠️ The tool **get_peak_ndvi** was used. It analyzes time-series NDVI data over the year to determine the maximum vegetation health, useful for crop monitoring or green cover studies."
    ),
    "get_solar_irradiance": (
        "🛠️ The tool **get_solar_irradiance** was used. It computes the average solar radiation received over the year using MODIS data, helping identify areas best suited for solar panels."
    ),
    "get_land_cover": (
        "🛠️ The tool **get_land_cover** was used. It retrieves detailed land cover categories (e.g., forest, urban, agriculture) from ESA’s WorldCover dataset at 10m resolution."
    )
}

    tool_explanation = tool_explanation_map.get(tool_name, "🛠️ A specific geospatial tool was used to process satellite data for this analysis.")

    reasoning_steps = [
    f"🧾 The system received your query: **\"{query}\"**.",
    f"🔍 It detected **{matched_city}** as the city mentioned in your question.",
    f"📌 The center of the city was used as the starting point for defining the study area.",
    f"📏 A circular region of **{buffer_km} km** around the city was selected to perform the analysis.",
    f"🤖 The AI agent interpreted the query and selected the geospatial tool: **{tool_name}**.",
    tool_explanation,
    f"📅 It pulled data from the year **2023**, covering your selected region in both space and time.",
    f"⚙️ The system then processed this data using logic specific to the task — such as slope filtering, spectral band ratios, or threshold-based classification.",
    f"🎨 A visualization was created using a color-coded overlay, so that different values or categories are easy to understand at a glance.",
    f"🗺️ The map shows the analysis result overlaid on the region — for example, areas at flood risk, green cover status, water presence, or solar energy potential.",
    f"✅ This gives you a complete spatial insight based on real satellite data, processed intelligently by the AI assistant."
    ]


    return layer, tool_label, response, reasoning_steps
