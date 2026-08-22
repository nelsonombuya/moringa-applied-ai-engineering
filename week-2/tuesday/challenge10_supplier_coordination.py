#  challenge_supplier_coordination.py
# TODO: Establish standard tool imports and environment configuration
import traceback
from os import environ

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# X TODO: Step 1 - Define determine_storage_facility tool
# If item_category is "cold-chain items", route to "Refrigerated Depot - Section X"
# Else route to "General Inventory Depot - Section Y"
@tool
def determine_storage_facility(item_category: str) -> str:
    """
    Determine the storage facility for a supplier item category.

    Cold-chain items are routed to the refrigerated depot; all other items
    are routed to the general inventory depot.

    Args:
        item_category: The category of the supplier item.

    Returns:
        str: The name of the storage facility where the item should be delivered.

    Example:
        >>> determine_storage_facility("cold-chain items")
        'Refrigerated Depot - Section X'
        >>> determine_storage_facility("general supplies")
        'General Inventory Depot - Section Y'
    """
    if item_category.strip().lower() == "cold-chain items":
        return "Refrigerated Depot - Section X"
    return "General Inventory Depot - Section Y"


# TODO: Step 2 - Define lookup_inventory_representative tool
# Match "section x" to "Representative Mwangi", match "section y" to
# "Representative Otieno"
@tool
def lookup_inventory_representative(storage_section: str) -> str:
    """
    Look up the inventory representative responsible for a storage section.

    Use this after determining the storage facility. Section X is handled by
    Representative Mwangi and Section Y by Representative Otieno.

    Args:
        storage_section: The section of the storage facility.

    Returns:
        str: The name of the inventory representative for the specified section.

    Example:
        >>> lookup_inventory_representative("Section X")
        'Representative Mwangi'
        >>> lookup_inventory_representative("Section Y")
        'Representative Otieno'
    """
    section = storage_section.strip().lower()
    if "section x" in section:
        return "Representative Mwangi"
    if "section y" in section:
        return "Representative Otieno"
    return "No inventory representative found for that storage section."


agent = create_agent(
    model=ChatOpenAI(
        temperature=0.0,
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=environ["OPENROUTER_API_KEY"],  # type:ignore
    ),
    tools=[
        determine_storage_facility,
        lookup_inventory_representative,
    ],
    system_prompt=(
        "You coordinate supplier procurement requests. Determine the correct "
        "storage facility first, then use the facility's section to find the "
        "inventory representative. Present a concise, clean result."
    ),
)


if __name__ == "__main__":
    query = (
        "A supplier needs to deliver cold-chain items. Which storage facility "
        "should receive them, and which inventory representative should handle "
        "the request?"
    )
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(result["messages"][-1].content)
    except Exception as e:  # noqa: F841
        print("An error occurred during agent execution: {e}")
        traceback.print_exc()
