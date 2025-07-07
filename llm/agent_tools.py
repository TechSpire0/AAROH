from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os
import ee

from gee.flood_gee import get_flood_mask, get_ndvi_mask, get_population_overlay

# Load environment variables
load_dotenv()

# Set up LLM using Together AI (Mistral)
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.0,  # Less hallucination
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1"
)

# Define tools
def get_tools(region):
    return [
        Tool.from_function(
            name="get_flood_mask",
            func=lambda q: get_flood_mask(region),
            description="Use this to detect flood-prone terrain using elevation < 200m and slope < 5 degrees."
        ),
        Tool.from_function(
            name="get_ndvi_mask",
            func=lambda q: get_ndvi_mask(region),
            description="Use this to find areas with low vegetation using NDVI < 0.2 from Sentinel-2."
        ),
        Tool.from_function(
            name="get_population_overlay",
            func=lambda q: get_population_overlay(region),
            description="Use this to get population density overlay using WorldPop data."
        )
    ]

# Main agent execution
def get_agent_layer(query, coords, buffer_km=30):
    region = ee.Geometry.Point(coords).buffer(buffer_km * 1000)
    tools = get_tools(region)

    # Define system prompt with strict format instruction
    system_message = SystemMessage(
        content=(
            "You are a spatial analysis assistant that uses Earth Engine tools to answer geographic queries. "
            "Use ONLY the provided tools to answer the query.\n\n"
            "Follow this format exactly:\n"
            "Action: <tool_name>\n"
            "Action Input: <query>\n"
            "Final Answer: <summary of what you found>\n\n"
            "Only use one tool that best answers the query. Do not generate Earth Engine code directly."
        )
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        max_iterations=4,
        max_execution_time=60,
        system_message=system_message
    )

    try:
        # Run the agent
        response = agent.run(query)
    except OutputParserException as e:
        response = str(e)
        tool_used = "Flood-Prone Terrain (Fallback)"
        layer = get_flood_mask(region)
        return layer, tool_used, response

    # Manual matching of tool used
    if "get_flood_mask" in response:
        layer = get_flood_mask(region)
        tool_used = "Flood-Prone Terrain"
    elif "get_ndvi_mask" in response:
        layer = get_ndvi_mask(region)
        tool_used = "Low Vegetation Zones"
    elif "get_population_overlay" in response:
        layer = get_population_overlay(region)
        tool_used = "Population Density"
    else:
         layer = get_ndvi_mask(region)
         tool_used = "Low Vegetation Zones"

    return layer, tool_used, response