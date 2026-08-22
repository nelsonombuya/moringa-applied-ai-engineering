# challenge8_depreciation.py
# Core skill (from Lab 9): a validated @tool wired into an AgentExecutor.
import traceback
from os import environ

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def calculate_asset_depreciation(
    initial_cost: float, salvage_value: float, useful_life_years: int
) -> str:
    """
    Calculates annual straight-line depreciation for a clinic asset.

    Args:
        initial_cost: The initial purchase cost of the asset.
        salvage_value: The estimated value of the asset at the end of its useful life.
        useful_life_years: The expected useful life of the asset in years.

    Returns:
        A string describing the annual depreciation expense, or an error message if inputs
            are invalid.
    """
    # Recall how Lab 9 guarded its division before computing.
    # Apply the SAME defensive idea here: what input would make this calculation invalid?
    # Return a clean error string for that case; otherwise return the depreciation.
    if useful_life_years <= 0:
        return "Error: Useful life must be greater than zero."

    annual_depreciation = (initial_cost - salvage_value) / useful_life_years
    return (
        f"The annual straight-line depreciation expense is: {annual_depreciation:.2f}"
    )


agent = create_agent(
    model=ChatOpenAI(
        temperature=0.0,
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=environ["OPENROUTER_API_KEY"],  # type:ignore
    ),
    tools=[calculate_asset_depreciation],
    system_prompt="""
    You are an AfyaPlus logistics assistant.
    You can calculate the annual straight-line depreciation of clinic assets based on
    their initial cost, salvage value, and useful life in years.
    """,
)


# Wire the tool into an agent and executor exactly as you did in Lab 9, then invoke
# with a depreciation question of your choice.
try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """
                    What is the annual straight-line depreciation of a clinic asset that
                    costs $10,000, has a salvage value of $2,000, and a useful life of 5
                    years?
                    """,
                }
            ]
        }
    )
    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)
except Exception as e:  # noqa: BLE001
    print(f"Error running agent: {e}")
    traceback.print_exc()
