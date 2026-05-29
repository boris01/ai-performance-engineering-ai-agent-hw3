import os
import sqlite3
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from src.domain import RouteClassification
from src.config import get_routing_engine, get_agent_reasoning_engine
from src.tools import ACTIVE_SESSION_ID


# Instantiate engines globally
router_model = get_routing_engine()
reasoning_agent_model = get_agent_reasoning_engine()


# =================================================================
# 1. DEFINE THE GRAPH STATE
# =================================================================
class AgentState(TypedDict):
    # 'add_messages' ensures LangGraph appends new messages natively
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =================================================================
# 2. DEFINE THE NODES (Actions)
# =================================================================
def agent_node(state: AgentState, tools: list):
    """The main reasoning engine that decides to use tools or answer the user."""
    messages = state["messages"]

    # Load the persistent user profile
    profile_filename = f"context_{ACTIVE_SESSION_ID}.md"
    user_profile_data = "No profile facts recorded yet."

    if os.path.exists(profile_filename):
        with open(profile_filename, "r") as f:
            user_profile_data = f.read().strip()

    reasoning_system_prompt = SystemMessage(content=(
        "ROLE & CONTEXT:\n"
        "You are an elite customer service data analyst agent handling database logs.\n"
        "Your mission is to formulate precise, data-driven insights using your registered analytical tools.\n\n"
        "OPERATIONAL INSTRUCTIONS:\n"
        "1. Always invoke your data tools to fetch or sample logs before answering a metric question. Never guess database content.\n"
        "2. If multiple tool calls are required to complete a complex query, run them sequentially.\n"
        "3. CRITICAL - CONVERSATION CONTEXT RESOLUTION: If the user asks a follow-up using vague terms (e.g., 'the last two', 'combine them', 'what about the other one'), you MUST look at your immediate conversation history to understand what they are referring to. Do not assume they mean the bottom of a dataset list.\n"
        "4. Present your findings back to the user in a clean, professional, and well-structured markdown layout.\n"
        "5. Remain concise and factual. Do not output text describing your internal function processing mechanics.\n"
        "6. CRITICAL: If asked for summary metadata, unique values, or a list of categories, use the 'get_category_metrics' tool EXACTLY ONCE. Do not systematically sample rows to find unique values manually.\n"
        "7. CRITICAL: If you learn something new about the user (e.g., their name, job, or preferences), use the 'save_user_fact' tool to record it.\n\n"

        "=========================================\n"
        "💾 PERSISTENT USER PROFILE FACTS:\n"
        f"{user_profile_data}\n"
        "=========================================\n"
        "Note: Reference these facts if the user asks what you remember about them or if it helps tailor your response."
    ))

    # Bind tools and invoke
    model_with_tools = reasoning_agent_model.bind_tools(tools)
    response = model_with_tools.invoke([reasoning_system_prompt] + messages)

    # Return a dictionary targeting the 'messages' key in the state
    return {"messages": [response]}


def out_of_scope_node(state: AgentState):
    """Handles boundary violations gracefully."""
    return {"messages": [AIMessage(
        content="Boundary Guard Notice: I am configured exclusively to interact with or answer questions regarding the historic customer service logs dataset.")]}


# =================================================================
# 3. DEFINE CONDITIONAL EDGES (Routing logic)
# =================================================================
def route_query(state: AgentState) -> str:
    """Gatekeeper router evaluating the active user intent."""
    messages = state["messages"]
    active_user_query = messages[-1].content

    history_context = ""
    if len(messages) > 1:
        history_context = "RECENT CONVERSATION HISTORY:\n"
        for msg in messages[-3:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_context += f"- {role}: {msg.content}\n"
        history_context += "\n"

    system_instructions = (
        "ROLE:\n"
        "You are an AI routing coordinator managing data access pathways for a database of e-commerce logs.\n"
        "CLASSIFY INTO EXACTLY ONE CATEGORY: 'structured', 'unstructured', or 'out_of_scope'.\n\n"
        "ROUTING PATHWAYS:\n"
        "1. 'structured': Any query requiring database interaction (tallies, rows, counts).\n"
        "   * CONTEXT RULE: Affirmations like 'yes', 'sure', 'show me' are 'structured'.*\n"
        "2. 'unstructured': Qualitative summaries, text sentiment requests, OR if the user is introducing themselves, stating their name, role, or business preferences.\n"
        "3. 'out_of_scope': ONLY if the query is entirely external to the system (e.g., writing python scripts, trivia).\n\n"
        "OUTPUT REQUIREMENT:\n"
        "Respond with a valid JSON object: {\"query_type\": \"structured\" | \"unstructured\" | \"out_of_scope\"}"
    )

    try:
        structured_router = router_model.with_structured_output(RouteClassification)
        decision = structured_router.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"{history_context}CURRENT USER INPUT TO CLASSIFY: {active_user_query}")
        ])

        # We route directly to the node names
        if decision.query_type == "out_of_scope":
            return "out_of_scope"
        return "agent" # both structured and unstructured request will go the agent, it is capable of handling both with its tools and reasoning

    except Exception as e:
        print(f"🚨 [System Guard Alert]: Structured output parsing failure: {str(e)}")
        return "agent"


# =================================================================
# 4. COMPILE THE GRAPH
# =================================================================
_conn = sqlite3.connect("state_memory.db", check_same_thread=False)
_db_checkpointer = SqliteSaver(_conn)
_db_checkpointer.setup()


def build_graph_agent(tools: list):
    """Compiles the LangGraph StateGraph API agent pipeline."""

    # Initialize the graph builder with our TypedDict state
    builder = StateGraph(AgentState)

    # Wrap the agent node so it has access to the tools passed in via dependency injection
    def wrapped_agent_node(state: AgentState):
        return agent_node(state, tools)

    # LangGraph Prebuilt ToolNode automatically executes Python functions correctly
    tool_node = ToolNode(tools)

    # 1. Add Nodes
    builder.add_node("agent", wrapped_agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("out_of_scope", out_of_scope_node)

    # 2. Add Conditional Entry Point (The Router)
    builder.add_conditional_edges(
        START,
        route_query,
        {
            "out_of_scope": "out_of_scope",
            "agent": "agent"
        }
    )

    # 3. Add ReAct Loop Logic
    # `tools_condition` is a native LangGraph function that checks if the agent returned a tool_call.
    # If yes, it routes to "tools". If no, it routes to "__end__".
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    # 4. Add standard edges
    builder.add_edge("tools", "agent")  # After tools finish, go back to the agent to evaluate
    builder.add_edge("out_of_scope", END)  # End immediately after an out-of-scope warning

    # 5. Compile with persistent memory
    customer_service_analyst_agent = builder.compile(checkpointer=_db_checkpointer)

    return customer_service_analyst_agent
