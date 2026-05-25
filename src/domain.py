from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field

class DatasetRow(BaseModel):
    """The row model matching the 5 physical columns in the Bitext CSV file."""
    flags: str
    instruction: str  # The customer query
    category: str     # The high-level semantic category for the intent (e.g. ACCOUNT)
    intent: str       # The intent corresponding to the user instruction tag (e.g. cancel_order)
    response: str     # The agent reply

class RouteClassification(BaseModel):
    """Structured classification output validation for the router node."""
    query_type: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="Classify whether the user query is a precise data question, an open-ended description, or completely unrelated."
    )