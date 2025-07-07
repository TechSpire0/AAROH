from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os
import ee
import streamlit as st

from gee.flood_gee import (
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
def get_agent_layer(query, coords, buffer_km=30):
    region = ee.Geometry.Point(coords).buffer(buffer_km * 1000)
    tools = get_tools(region)

    system_message = SystemMessage(
        content=(
            "You are a spatial analysis assistant that uses Earth Engine tools to answer geographic queries. "
            "Use ONLY the provided tools to answer the query.\n\n"
            "Format:\n"
            "Action: <tool_name>\n"
            "Action Input: <query>\n"
            "Final Answer: <summary>\n\n"
            "Only use one tool that best answers the query."
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
        return layer, "Flood-Prone Terrain (Fallback)", response

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

    return layer, tool_label, response
