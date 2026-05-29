import os
import sqlite3
from typing import Sequence, Any
from langgraph.func import entrypoint, task
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.domain import RouteClassification
from src.config import get_routing_engine, get_agent_reasoning_engine
from src.tools import ACTIVE_SESSION_ID

router_model = get_routing_engine()
reasoning_agent_model = get_agent_reasoning_engine()


@task
def determine_query_route(messages: Sequence[BaseMessage]) -> str:
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
        "You are an AI routing coordinator managing data access pathways for a database of e-commerce customer service logs.\n"
        "Your task is to analyze the user's latest request and classify the query destination into exactly one category: 'structured', 'unstructured', or 'out_of_scope'.\n\n"

        "DATABASE SCHEMA CONTEXT:\n"
        "The database contains raw communication logs organized into category partitions such as: SHIPPING, DELIVERY, ORDER, REFUNDS, and ACCOUNT.\n\n"

        "ROUTING PATHWAYS:\n"
        "1. 'structured': Select this for any query requiring database interaction (tallies, rows, counts, example lookups).\n"
        "   * CONTEXT RULE: If the user says a short affirmation like 'yes', 'sure', 'go ahead', or 'show me', classify it as 'structured'.*\n\n"

        "2. 'unstructured': Select this for qualitative summaries, text sentiment requests, OR if the user is introducing themselves, stating their name, role, or business preferences. (This allows the agent to save their profile).\n\n"

        "3. 'out_of_scope': Select this ONLY if the query is entirely external to the system (e.g., writing python scripts, poems, trivia). Do NOT select this for user introductions.\n\n"

        "OUTPUT REQUIREMENT:\n"
        "Respond with a valid JSON object matching this schema exactly:\n"
        "{\n"
        "  \"query_type\": \"structured\" | \"unstructured\" | \"out_of_scope\"\n"
        "}\n"
        "Output only the raw JSON string. No markdown wrappers."
    )

    try:
        structured_router = router_model.with_structured_output(RouteClassification)
        decision = structured_router.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"{history_context}CURRENT USER INPUT TO CLASSIFY: {active_user_query}")
        ])
        return decision.query_type
    except Exception as e:
        print(f"🚨 [System Guard Alert]: Structured output parsing failure: {str(e)}")
        return "structured"


@task
def execute_react_reasoning(messages: Sequence[BaseMessage], tools: list) -> BaseMessage:
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

    full_payload = [reasoning_system_prompt] + list(messages)
    model_with_tools = reasoning_agent_model.bind_tools(tools)
    return model_with_tools.invoke(full_payload)


@task
def execute_single_tool(tool_call: dict, tools_list: list) -> ToolMessage:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]

    target_tool = next((t for t in tools_list if t.name == tool_name), None)

    if not target_tool:
        output_content = f"Execution Fault: Tool '{tool_name}' is not registered."
    else:
        try:
            output_content = target_tool.invoke(tool_args)
        except Exception as e:
            output_content = f"Runtime error inside tool: {str(e)}"

    return ToolMessage(content=str(output_content), tool_call_id=tool_id, name=tool_name)


@task
def handle_out_of_scope() -> AIMessage:
    return AIMessage(
        content="Boundary Guard Notice: I am configured exclusively to interact with or answer questions regarding the historic customer service logs dataset.")


@task
def handle_loop_exhaustion(max_iteration) -> AIMessage:
    return AIMessage(
        content=f"Processing Limit Notice: The computation exceeded the maximum safe execution iterations threshold ({max_iteration} steps).")


_conn = sqlite3.connect("state_memory.db", check_same_thread=False)
_db_checkpointer = SqliteSaver(_conn)


def build_agent(tools: list):
    @entrypoint(checkpointer=_db_checkpointer)
    def customer_service_analyst_agent(messages: Sequence[BaseMessage], *, previous: Any = None) -> Sequence[
        BaseMessage]:

        # Merge the natively injected history with the single fresh input message
        conversation_history = list(previous) if previous else []
        conversation_history.extend(list(messages))

        query_type = determine_query_route(conversation_history).result()
        print(f"DEBUG: Selected routing pathway identified as -> [{query_type}]")

        if query_type == "out_of_scope":
            decline_msg = handle_out_of_scope().result()
            return [decline_msg]
        # both structured and unstructured request will go the agent, it is capable of handling both with its tools and reasoning
        max_iterations = 12
        for iteration in range(max_iterations):
            agent_response = execute_react_reasoning(conversation_history, tools).result()
            conversation_history.append(agent_response)

            if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
                print(f"⚙️ [ReAct Step]: Model invoking {len(agent_response.tool_calls)} processing tools...")
                for call in agent_response.tool_calls:
                    print(f"   🛠️ [Tool Invocation]: {call['name']}()")
                    print(f"   📥 [Arguments Passed]: {call['args']}\n")
                    tool_message_result = execute_single_tool(call, tools).result()
                    conversation_history.append(tool_message_result)
                continue

            return conversation_history

        fallback_msg = handle_loop_exhaustion(max_iterations).result()
        return [fallback_msg]

    return customer_service_analyst_agent