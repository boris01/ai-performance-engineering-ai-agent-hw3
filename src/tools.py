from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.interfaces import IBitextRepository


class ToolFilterInput(BaseModel):
    category: Optional[str] = Field(None,
                                    description="Category string filter flag (e.g. 'ACCOUNT', 'SHIPPING'). Exact uppercase match.")
    intent: Optional[str] = Field(None,
                                  description="Intent identifier tag label string (e.g. 'get_refund', 'cancel_order'). Lowercase.")


class ToolSampleInput(ToolFilterInput):
    limit: int = Field(3, description="Total record lines count threshold ceiling size to extract from the log vaults.")


class ToolDistributionInput(BaseModel):
    category: Optional[str] = Field(None,
                                    description="The parent cluster category to break down downstream intent values frequencies for.")

ACTIVE_SESSION_ID = "default"

def set_active_session(session_id: str):
    global ACTIVE_SESSION_ID
    ACTIVE_SESSION_ID = session_id
def build_computational_tools(repo: IBitextRepository) -> list:
    """Dependency injection factory compiling tools using abstract interface inputs."""

    @tool(args_schema=ToolFilterInput)
    def count_customer_service_logs(category: Optional[str] = None, intent: Optional[str] = None) -> str:
        """Calculates the exact total count of customer support logs matching specific criteria."""
        total = repo.count_records(category, intent)
        return f"Tally Result: Found exactly {total} operational database items matching filters."

    @tool(args_schema=ToolDistributionInput)
    def calculate_intent_distribution(category: Optional[str] = None) -> str:
        """Computes a structural breakdown table displaying distinct user intent value counts."""
        distribution_map = repo.get_distribution(category)
        if not distribution_map:
            return "Analytics Alert: The requested domain context yielded zero matching metrics profiles."
        lines = ["| Intent Identifier | Occurrence Count |", "| :--- | :--- |"]
        for key, count in distribution_map.items():
            lines.append(f"| {key} | {count} |")
        return "\n".join(lines)

    @tool(args_schema=ToolSampleInput)
    def get_customer_service_examples(category: Optional[str] = None, intent: Optional[str] = None,
                                      limit: int = 3) -> str:
        """Retrieves direct logs showing the customer message (instruction) and how the agent replied (response)."""
        matching_rows = repo.get_row_samples(category, intent, limit)
        if not matching_rows:
            return "Query Warning: No raw transcript text items matched the specified criteria parameters."

        output_blocks = []
        for rank, row in enumerate(matching_rows):
            output_blocks.append(
                f"### Record Fragment #{rank + 1}\n"
                f"* **Customer Instruction:** \"{row.instruction}\"\n"
                f"* **Agent System Response:** \"{row.response}\"\n"
                "---"
            )
        return "\n".join(output_blocks)

    @tool
    def sample_category_records(category_name: str, limit: int = 5) -> str:
        """
        Fetches raw example text logs matching a specific system category bucket.
        Use this tool whenever the user asks to see 'examples', 'samples', or 'records'
        of a specific category partition (e.g., SHIPPING, DELIVERY, ORDER).
        """
        sample_output = repo.sample_category_records(category=category_name, limit=limit)
        return sample_output if isinstance(sample_output, str) else "Unexpected output format from sample_category_records tool."

    @tool
    def get_category_metrics() -> str:
        """
        Scans the entire customer service log dataset and returns a complete list
        of all unique category names present along with their distribution totals.
        Use this tool when users ask 'what categories exist', 'list the categories',
        or ask for a structural summary of the dataset.
        """
        return repo.get_category_metrics()



    @tool
    def save_user_fact(fact: str) -> str:
        """
        CRITICAL: Use this tool to save important facts about the user (e.g., their name, preferences, or topics they frequently ask about).
        Call this tool whenever you learn a new, persistent detail about the user. Do NOT use this for a temporary conversation context.
        """
        filename = f"context_{ACTIVE_SESSION_ID}.md"

        # Append the distilled fact to the user's specific Markdown file
        with open(filename, "a") as f:
            f.write(f"- {fact}\n")

        return f"Successfully saved fact to user profile: {fact}"

    return [count_customer_service_logs, calculate_intent_distribution, get_customer_service_examples, sample_category_records, get_category_metrics, save_user_fact]