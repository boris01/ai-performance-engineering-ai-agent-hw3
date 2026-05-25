import os
from mcp.server.fastmcp import FastMCP

from src import PandasBitextRepository, build_computational_tools

csv_file_path = os.getenv("DATA_PATH", "data/bitext_customer_service.csv")
repo = PandasBitextRepository(csv_filepath=csv_file_path)

langchain_tools = build_computational_tools(repo=repo)

tools_dict = {tool.name: tool for tool in langchain_tools}

mcp = FastMCP("CustomerServiceMCPServer")

@mcp.tool()
def count_logs(intent: str) -> str:
    """Calculates the exact total count of customer support logs matching specific criteria."""
    return tools_dict["count_customer_service_logs"].invoke({"intent": intent})

@mcp.tool()
def get_metrics() -> str:
    """
    Scans the entire customer service log dataset and returns a complete list
    of all unique category names present along with their distribution totals.
    Use this tool when users ask 'what categories exist', 'list the categories',
    or ask for a structural summary of the dataset.
    """
    return tools_dict["get_category_metrics"].invoke({})

@mcp.tool()
def fetch_samples(category: str, limit: int = 5) -> str:
    """
   Fetches raw example text logs matching a specific system category bucket.
   Use this tool whenever the user asks to see 'examples', 'samples', or 'records'
   of a specific category partition (e.g., SHIPPING, DELIVERY, ORDER).
   """
    return tools_dict["sample_category_records"].invoke({"category": category, "limit": limit})


if __name__ == "__main__":
    # Start the server
    mcp.run()