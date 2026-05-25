import os
import argparse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from src import PandasBitextRepository, build_computational_tools, set_active_session
from src.graph_api import build_graph_agent


def launch_modern_cli():
    parser = argparse.ArgumentParser(description="Modern LangGraph Agent Terminal with Persistent Memory")
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Unique session ID thread token string to persist or recover history states"
    )
    args = parser.parse_args()

    # 1. Initialize session and tools
    set_active_session(args.session)
    csv_file_path = os.getenv("DATA_PATH", "data/bitext_customer_service.csv")
    storage_provider = PandasBitextRepository(csv_filepath=csv_file_path)
    executable_tools = build_computational_tools(repo=storage_provider)

    # 2. Build the Graph API Agent
    customer_service_analyst_agent = build_graph_agent(tools=executable_tools)

    # 3. Configure recursion and threads
    graph_config = {
        "configurable": {"thread_id": args.session},
        "recursion_limit": 12
    }

    print("=" * 65)
    print("🚀 MODERN LANGGRAPH AGENT TERMINAL ACTIVE")
    print(f"  Orchestration Thread ID Session Key: {args.session}")
    print("  Memory Driver: SQLite Database Persistence Layer Active")
    print("=" * 65)

    while True:
        try:
            user_string = input("\nYou: ").strip()
            if user_string.lower() in ["exit", "quit"]:
                break
            if not user_string:
                continue

            # StateGraph requires a dictionary payload targeting the "messages" key
            input_payload = {"messages": [HumanMessage(content=user_string)]}
            final_output = None

            print("\n⚙️ [Processing Multi-Step Workflow Trace]:")

            # 4. FORCE stream_mode="values" to guarantee standard state dictionary outputs
            for event in customer_service_analyst_agent.stream(input_payload, config=graph_config,
                                                               stream_mode="values"):

                # Check if the state contains our messages array
                if "messages" in event:
                    final_output = event["messages"]

                    # Grab the very last message that was just added to the state
                    last_msg = final_output[-1]

                    # Rebuild the UI trace by inspecting what kind of message it is
                    if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for call in last_msg.tool_calls:
                            print(f"    🤖 Agent Trace -> Decided to call: '{call['name']}'")

                    elif last_msg.type == "tool":
                        print(f"  ✦ [Completed Task Step]: '{last_msg.name}' tool executed")

            # 5. Output the final string
            if final_output and len(final_output) > 0:
                print(f"\n📢 Final Agent Response:\n{final_output[-1].content}")
            else:
                print("\n📢 Final Agent Response: [Error: Checkpoint thread tracking context sync failure]")

        # 6. Catch infinite loops
        except GraphRecursionError:
            print(
                f"\n📢 Final Agent Response:\nProcessing Limit Notice: The computation exceeded the maximum safe execution iterations threshold ({graph_config['recursion_limit']} steps).")

        # 7. Catch Ctrl+C to exit gracefully
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    launch_modern_cli()