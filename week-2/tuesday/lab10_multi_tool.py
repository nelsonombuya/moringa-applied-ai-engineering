# lab10_multi_tool.py
import traceback
from os import environ

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Returns the current stock count for a medication at the AfyaPlus clinic."""
    stock = {"amoxicillin": 120, "paracetamol": 540}
    return f"{stock.get(medication_name.lower(), 0)} units"


@tool
def calculate_shift_cost(hours: float, hourly_rate: float) -> str:
    """Calculates the total cost of a staff shift."""
    if hours < 0 or hourly_rate < 0:
        return "Error: hours and rate must be non-negative."
    return f"Shift cost: {hours * hourly_rate:.2f}"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AfyaPlus operations assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_agent(
    model=ChatOpenAI(
        temperature=0.0,
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=environ["OPENROUTER_API_KEY"],  # type:ignore
    ),
    tools=[get_clinic_stock_count, calculate_shift_cost],
    system_prompt="You are an AfyaPlus operations assistant.",
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How much paracetamol do we have, and what is the cost of an 8-hour shift at 450 per hour?",
                }
            ]
        }
    )
    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)
except Exception as e:  # noqa: BLE001
    print(f"Error running agent: {e}")
    traceback.print_exc()
