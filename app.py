````python
import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Render provides this through Environment Variables.
# Do NOT hardcode the API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemma-4-31b-it"
)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return stdout or
    an error traceback.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate specific test scenarios for a coding task.
    """

    prompt = (
        "You are a Senior QA Engineer.\n\n"
        "Generate 3 to 5 highly specific test scenarios "
        "for the following coding task:\n\n"
        f"{task_description}\n\n"
        "Include both standard cases and edge cases.\n"
        "Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(
                    str(item.get("text", ""))
                )
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# ============================================================
# 4. GRAPH NODES
# ============================================================

def developer_node(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this coding task:\n\n"
        f"{task}\n\n"
        "Requirements:\n"
        "- Return ONLY executable Python code.\n"
        "- Do not include markdown.\n"
        "- Do not include ```python.\n"
        "- Do not include explanations."
    )

    response = llm.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(
                    str(item.get("text", ""))
                )
            else:
                parts.append(str(item))

        code_str = "\n".join(parts)

    else:
        code_str = str(content)

    return {
        "code": code_str.strip()
    }


def tester_node(state: CrewState):

    task = state["messages"][-1].content

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = (
        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED\n\n"
        f"{test_cases}"
    )

    return {
        "report": report
    }


# ============================================================
# 5. LANGGRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    END
)

rt_app = workflow.compile()


# ============================================================
# 6. LANGSERVE INPUT / OUTPUT ADAPTER
# ============================================================

class AgentInput(TypedDict):
    input: str


def format_for_agent(x) -> dict:
    """
    Convert LangServe Playground input into
    the LangGraph state expected by the workflow.
    """

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "next_step": None,
        "code": None,
        "report": None,
    }


def extract_agent_response(result: dict) -> dict:
    """
    Return a clean response containing the generated
    code and final report.
    """

    return {
        "generated_code": result.get(
            "code",
            ""
        ),
        "report": result.get(
            "report",
            "No report generated."
        )
    }


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | rt_app
    | RunnableLambda(extract_agent_response)
)


# ============================================================
# 7. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="AI Coding Crew",
    description="LangGraph + Gemini coding task pipeline",
    version="1.0.0",
)


# This creates:
#
# /agent/invoke
# /agent/batch
# /agent/stream
# /agent/playground/
#
add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
)


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Coding Crew",
        "model": MODEL_NAME,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# 9. LOCAL / RENDER STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
````
