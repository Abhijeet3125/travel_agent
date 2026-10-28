from langchain_core.tools import tool

@tool
def submit_trip_preferences(
    destination: str,
    start_date: str,
    end_date: str,
    budget: str,
    travelers: int,
    interests: str
) -> str:
    """Call this tool ONLY when you have successfully collected all 6 mandatory trip details."""

    return """Preferences saved. Handoff to planner"""

