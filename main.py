import os
import argparse
from langchain_core.messages import HumanMessage
from src import PandasBitextRepository, build_computational_tools, set_active_session

from src.graph import build_agent


def launch_modern_cli():
    parser = argparse.ArgumentParser(description="Modern LangGraph Agent Terminal with Persistent Memory")
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Unique session ID thread token string to persist or recover history states"
    )
    args = parser.parse_args()

    set_active_session(args.session)

    csv_file_path = os.getenv("DATA_PATH", "data/bitext_customer_service.csv")
    storage_provider = PandasBitextRepository(csv_filepath=csv_file_path)
    executable_tools = build_computational_tools(repo=storage_provider)

    customer_service_analyst_agent = build_agent(tools=executable_tools)

    graph_config = {"configurable": {"thread_id": args.session}}

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

            input_payload = [HumanMessage(content=user_string)]

            final_output = None
            print("\n⚙️ [Processing Multi-Step Workflow Trace]:")

            for event in customer_service_analyst_agent.stream(input_payload, config=graph_config):
                if isinstance(event, dict) and "customer_service_analyst_agent" in event:
                    final_output = event["customer_service_analyst_agent"]

                if "task" in event:
                    task_name = event["task"]
                    task_data = event.get("data", {})
                    print(f"  ✦ [Completed Task Step]: {task_name}")

                    if task_name == "execute_react_reasoning" and hasattr(task_data, "tool_calls"):
                        if task_data.tool_calls:
                            for call in task_data.tool_calls:
                                print(f"    🤖 Agent Trace -> Decided to call: '{call['name']}'")

            if final_output and len(final_output) > 0:
                print(f"\n📢 Final Agent Response:\n{final_output[-1].content}")
            else:
                print("\n📢 Final Agent Response: [Error: Checkpoint thread tracking context sync failure]")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    launch_modern_cli()