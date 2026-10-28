from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: Annotated[int, operator.add]

    # intake agent output
    destination: str | None
    start_date: str | None
    end_date: str | None
    budget: str | None
    travelers: int | None
    interests: str | None

    #planner agent output
    deaft_itinerary: str | None

    #Critique agent
    validation_errors: list[str] | None
    revesion_count: Annotated[int, operator.add]




