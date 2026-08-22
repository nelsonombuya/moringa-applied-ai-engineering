# lab9_math_suite.py
import traceback
from os import environ

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    temperature=0.0,
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=environ["OPENROUTER_API_KEY"],  # type:ignore
)


@tool
def calculate_facility_utilization(active_beds: int, total_beds: int) -> str:
    """Calculates the capacity utilisation percentage for a clinic ward."""
    try:
        if total_beds <= 0:
            return "Error: total beds must be greater than zero."
        pct = (active_beds / total_beds) * 100
        return f"Utilisation: {pct:.1f}%"
    except Exception as e:  # noqa: BLE001
        return f"Error computing utilisation: {e} - {traceback.format_exc(limit=1)}"


tools = [calculate_facility_utilization]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are an AfyaPlus operations assistant.",
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "If 84 of our 100 beds are occupied, what is utilisation?",
                }
            ]
        }
    )
    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)
except Exception as e:  # noqa: BLE001
    print(f"Error running agent: {e}")
    traceback.print_exc()
