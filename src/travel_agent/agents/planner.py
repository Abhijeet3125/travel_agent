from src.travel_agent.llm import llm
from src.travel_agent.tools.weather import get_weather, get_coordinates
from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.travel_agent.schemas.state import MessagesState
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from IPython.display import Image, display
from datetime import datetime, timedelta


tools = [get_weather, get_coordinates]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)


today = datetime.now().date()
max_forecast_date = today + timedelta(days=16)

system_prompt = f"""
You are a helpful travel assistant tasked with planning trips.

CRITICAL WEATHER TOOL RULES:
1. Today's date is {today}.
2. The weather forecast tool can ONLY retrieve data for dates between today and {max_forecast_date} (up to 16 days in the future).
3. If the user requests weather for dates outside this exact window, DO NOT call the weather tool. Instead, politely inform the user that you can only provide forecasts up to 16 days in advance.
4. The date's specified by the user are for the current year unless speicified otherwise.
"""

def llm_call(state: dict):

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=system_prompt
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0)+1
    }


def tool_node(state: dict):
    """performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content = observation, tool_call_id = tool_call["id"]))

    return {"messages": result}



def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)


agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
 
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

agent.get_graph(xray=True).draw_mermaid_png(output_file_path="agent_graph.png")

messages = [HumanMessage(content="can i know the temperature for delhi from 23 october to 27 october since i am planning to travel there.")]
messages = agent.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()