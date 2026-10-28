from src.travel_agent.llm import llm
from src.travel_agent.schemas.state import MessagesState
from langchain.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from datetime import datetime
from src.travel_agent.tools.ask_user import ask_user
from src.travel_agent.tools.submit_trip_preferences import submit_trip_preferences
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

tools = [ask_user, submit_trip_preferences]
tools_by_name = {}

for tool in tools:
    tools_by_name[tool.name] = tool

model_with_tools = llm.bind_tools(tools)

today = datetime.now().date()

system_prompt = f"""
You are the Intake Specialist for an expert travel planning system. 
Your ONLY job is to greet the user, ask clarifying questions, and extract the exact constraints needed to plan a trip.

CURRENT DATE: {today} (Use this to calculate relative dates like "next week").

MANDATORY TRIP DETAILS TO COLLECT:
1. Destination (City and Country)
2. Start Date and End Date (Must be exact dates, e.g., YYYY-MM-DD)
3. Total Budget (Include the currency)
4. Number of Travelers
5. Trip Pace & Interests (e.g., fast-paced historical, relaxing foodie, outdoorsy)

CRITICAL RULES:
- DO NOT PLAN THE TRIP: You are strictly forbidden from generating itineraries, suggesting hotels, or listing activities. 
- BE CONVERSATIONAL: If the user is missing multiple details, you have to use the ask_user tool specifically, don't directly ask the question.
- VALIDATE: If a budget seems impossibly low for the destination, politely double-check with the user.

COMPLETION CRITERIA:
Once you have successfully collected ALL 5 mandatory details, you must stop asking questions. 
Instead, call the `submit_trip_preferences` tool with the extracted data so the Planning Department can take over.
"""

def llm_call(state: MessagesState):

    response = model_with_tools.invoke(
                    [SystemMessage(content=system_prompt)] + state["messages"]
                )
    return {
        "messages": [response],
        "llm_calls": 1
    }


def tool_node(state: dict):
    """performs the tool call"""

    result = []
    updated_state = {}

    for tool_call in state["messages"][-1].tool_calls:

        if tool_call["name"] == "ask_user":
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id = tool_call["id"]))

        elif tool_call["name"] == "submit_trip_preferences":
            args = tool_call["args"]
            updated_state["destination"] = args.get("destination")
            updated_state["start_date"] = args.get("start_date")
            updated_state["end_date"] = args.get("end_date")
            updated_state["budget"] = args.get("budget")
            updated_state["travelers"] = args.get("travelers")
            updated_state["interests"] = args.get("interests")

            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation , tool_call_id = tool_call["id"]))

        else:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content = observation, tool_call_id = tool_call["id"]))

    updated_state["messages"] = result
    return updated_state


def should_continue(state: dict) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    #if we dont get any tool call in last message then end
    if last_message.tool_calls:
        return "tool_node"

    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call");
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)


config: RunnableConfig = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="i am planning a trip to delhi from 23 oct to 27 oct")]

stream_input = {"messages": messages}

while True:
    stream = agent.stream_events(stream_input, config=config, version="v3")

    for message in stream.messages:
       for token in message.text:
           print(token, end="", flush=True)

    if not stream.interrupted:
        final_state = stream.output
        break

    ai_question = stream.interrupts[0].value
    user_response = input(f"\nAI: {ai_question}\n You: ")
    stream_input = Command(resume=user_response)


final_snapshot = agent.get_state(config)

# The actual dictionary of your variables lives in the .values attribute
state_dict = final_snapshot.values

print(f"messages:\n {state_dict.get("messages")}")
print(f"Destination: {state_dict.get('destination')}")
print(f"Start Date: {state_dict.get('start_date')}")
print(f"End Date: {state_dict.get('end_date')}")
print(f"Budget: {state_dict.get('budget')}")
print(f"Travelers: {state_dict.get('travelers')}")
print(f"Interests: {state_dict.get('interests')}")



