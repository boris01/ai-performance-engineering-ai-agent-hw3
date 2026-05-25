# Customer Service Data Analyst Agent 
**Students:** Boris Shterenberg & Gregory Jerusalemsky

## 🏗️ Architecture Overview
This project implements a professional, multi-step AI agent designed to analyze customer service logs. 

**Dual LangGraph Implementation (Bonus):**
To demonstrate a comprehensive understanding of the framework, this repository contains two distinct architectural implementations of the agent:
1. **StateGraph API (`main_graph.py` & `src/graph_api.py`):** The primary, production-grade implementation using LangGraph's node-and-edge architecture with native thread checking.
2. **Functional API (`main.py` & `src/graph.py`):** An alternative implementation using LangGraph's `@entrypoint` decorator and a manual ReAct loop.

**Core Agent Workflow:**
* **The Router:** Incoming queries are evaluated by a dedicated routing node/model. Out-of-scope prompts are immediately rejected by a boundary guard. Valid queries (structured and unstructured) are routed to the reasoning engine.
* **The Agent:** The core engine uses a ReAct pattern to dynamically invoke custom Python tools to interact with a Pandas-powered bitext dataset.
* **Persistent Memory:** Both graphs use LangGraph's `SqliteSaver` checkpointer to maintain conversational context across CLI restarts.
* **Episodic User Profile:** The agent has a `save_user_fact` tool that writes learned user facts to a local Markdown file, dynamically injecting them into the system prompt for personalized memory.

### 🧠 Model Choice (Nebius Token Factory)
We used a multi-agent architecture with specific models tailored to their operational roles, optimizing for both cost and capability:

* **Routing Engine:** We utilized `nvidia/Nemotron-3-Nano-Omni` for the router. Because routing is a static, single-turn classification task relying on structured outputs, this smaller, ultra-fast model was chosen as the most cost-effective and efficient engine for the job.
* **Reasoning Engine:** We utilized `nvidia/nemotron-3-super-120b-a12b` for the core agent. This massive 120-billion-parameter model was selected as the best engine for reliable function calling, complex data synthesis, and handling the multi-step reasoning chains required by the ReAct loop.

---

## 🛠️ Tool Arsenal
The agent has access to the following typed tools:
1. `count_customer_service_logs`: Calculates the exact total count of logs matching specific category and intent criteria.
2. `calculate_intent_distribution`: Computes a structural breakdown table displaying intent frequencies within a specific category.
3. `get_customer_service_examples`: Retrieves direct logs showing the exact customer instruction and the corresponding agent response.
4. `sample_category_records`: Fetches raw example text logs matching a specific system category bucket.
5. `get_category_metrics`: Scans the entire dataset to return a complete list of all unique categories and their distribution totals.
6. `save_user_fact`: Saves learned user preferences and details to a local Markdown file to build persistent episodic memory.

---

## 🚀 Setup & Execution

### 1. Installation
Ensure you have Python 3.13 installed. Clone this repository, activate your virtual environment, and install the dependencies:
```bash
pip install -r requirements.txt
```
*Note: Ensure your `.env` file is configured with your Nebius API keys (`NEBIUS_API_KEY` and `NEBIUS_BASE_URL`).*

### 2. Running the Agent (Interactive CLI)
Launch the agent by providing a unique session ID, which tracks your SQLite persistent memory thread.

**Option A: Run the StateGraph API (Recommended)**
```bash
python main_graph.py --session grading_stategraph_run
```

**Option B: Run the Functional API**
```bash
python main.py --session grading_functional_run
```

### 3. Example Queries to Test
Once the terminal is active, try pasting these exact queries to test the agent's routing, memory, and multi-step reasoning capabilities:
* **Structured Math:** *"How many refund requests did we get?"*
* **Unstructured Summarization:** *"Summarize how agents respond to complaint intents in the FEEDBACK category."*
* **Context/Memory Follow-up:** *"What is the distribution of intents for that category?"*
* **Out-of-Scope Guardrail:** *"Who won the 2024 Champions League?"*

---

## 🌐 MCP Server Integration
This project includes a fully functional Model Context Protocol (MCP) server (`server.py`) powered by FastMCP, exposing our core data tools to external AI clients.

### Starting the Server
You can start the server using the MCP inspector to test it via Web UI:
```bash
npx @modelcontextprotocol/inspector uv run server.py
```
Alternatively, you can run it silently for stdio connections:
```bash
python server.py
```

### Connecting an External Client
Here is a Python snippet demonstrating how an external client (or another agent) connects to the running MCP server to call the `get_metrics` tool natively over standard input/output:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Request metadata directly from the MCP Server
            result = await session.call_tool("get_metrics", arguments={})
            print(f"Server Response: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run_client())
```