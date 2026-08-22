# lab10_multi_tool.py
import traceback
from os import environ

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """
    Returns the current stock count for a medication at the AfyaPlus clinic.

    Args:
        medication_name: The name of the medication to check stock for.

    Returns:
        str: A string indicating the number of units in stock for the specified
            medication.

    Example:
        >>> get_clinic_stock_count("amoxicillin")
        '120 units'
    """
    stock = {"amoxicillin": 120, "paracetamol": 540}
    return f"{stock.get(medication_name.lower(), 0)} units"


@tool
def calculate_shift_cost(hours: float, hourly_rate: float) -> str:
    """
    Calculates the total cost of a staff shift.

    Args:
        hours: The number of hours worked in the shift.
        hourly_rate: The hourly rate of pay in currency units.

    Returns:
        str: A string indicating the total shift cost, or an error message if inputs
            are invalid.

    Example:
        >>> calculate_shift_cost(8, 450)
        'Shift cost: 3600.00'
    """
    if hours < 0 or hourly_rate < 0:
        return "Error: hours and rate must be non-negative."
    return f"Shift cost: {hours * hourly_rate:.2f}"


def calculate_asset_depreciation(
    initial_cost: float, salvage_value: float, useful_life_years: int
) -> str:
    """
    Calculate a clinic asset's annual straight-line depreciation expense.

    Args:
        initial_cost: The asset's original purchase cost.
        salvage_value: The asset's estimated value at the end of its useful life.
        useful_life_years: The asset's useful life in years; must be greater than zero.

    Returns:
        A formatted string containing the annual depreciation expense, or an error
        message when useful_life_years is invalid.

    Notes:
        The expense is calculated as (initial_cost - salvage_value) /
        useful_life_years.
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
    tools=[get_clinic_stock_count, calculate_shift_cost, calculate_asset_depreciation],
    system_prompt="You are an AfyaPlus operations assistant.",
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        """
                        Complete this AfyaPlus operations audit using every available
                        tool. Check the current stock of paracetamol, calculate the cost
                        of an 8-hour shift at 450 per hour, and calculate the annual
                        straight-line depreciation for an asset with an initial cost of
                        120000, a salvage value of 20000, and a useful life of 5 years.
                        Report each result clearly.
                        """
                    ),
                }
            ]
        }
    )
    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)
except Exception as e:  # noqa: BLE001
    print(f"Error running agent: {e}")
    traceback.print_exc()
