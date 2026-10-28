from langchain_core.tools import tool
from langgraph.types import interrupt

@tool 
def ask_user(question: str) -> str:
    """Use this tool to ask the user a direct question when you are mising required trip details."""

    answer = interrupt(question)
    return answer