# src/config.py
import os
from langchain_openai import ChatOpenAI

# 1. Load configuration variables once into memory
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

def get_routing_engine() -> ChatOpenAI:
    """The fastest, most cost-effective engine for static single-turn query routing."""
    return ChatOpenAI(
        model="nvidia/Nemotron-3-Nano-Omni",
        temperature=0.0,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL
    )

def get_agent_reasoning_engine() -> ChatOpenAI:
    """The high-capacity model for reliable function calling and multi-step reasoning chains."""
    return ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b",
        temperature=0.0,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL
    )