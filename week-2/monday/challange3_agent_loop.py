# challenge3_agent_loop.py
from os import environ

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Instantiate the reasoning brain
load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.0,
    api_key=environ.get("OPENROUTER_API_KEY"),  # type:ignore
    base_url="https://openrouter.ai/api/v1",
)


# X TODO: Step 1 - Complete this custom tool using the @tool decorator
@tool
def lookup_specialist_department(specialty_name: str) -> str:
    """
    Use this tool to find out the location or status of a specific medical specialty
    department at AfyaPlus.

    Args:
        specialty_name (str): The name of the medical specialty (e.g., "pediatrics",
            "cardiology", "dermatology").

    Returns:
        str: A string describing the location or status of the specialty department.

    Example:
        >>> lookup_specialist_department("pediatrics")
        'Located in Wing A, open until 8 PM.'
    """
    roster = {
        "pediatrics": "Located in Wing A, open until 8 PM.",
        "cardiology": "Located in Main Tower, requires pre-booking.",
        "dermatology": "Nairobi Hub clinic, fully booked this week.",
    }

    # X TODO: Write logic to normalize text and pull from the roster dict
    return roster.get(specialty_name.lower(), "Department not found.")


# X TODO: Create a tool for checking clinic stock count using the @tool decorator
@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """
    Queries the AfyaPlus central inventory system for a specific drug.
    Use this tool whenever a patient asks if a medicine is physically available in the
    pharmacy.

    Args:
        medication_name (str): The name of the medication to check stock for.

    Returns:
        str: A string indicating the stock status of the medication.

    Example:
        >>> get_clinic_stock_count("amoxicillin")
        '0 units left - currently backordered.'
    """
    # Mocking internal system database layer lookup
    inventory_db = {
        "paracetamol": "1,200 tablets available in Nairobi Hub.",
        "amoxicillin": "0 units left - currently backordered.",
        "antacid": "450 bottles in stock in Mombasa clinic.",
    }
    normalized_name = medication_name.lower().strip()
    return inventory_db.get(
        normalized_name,
        f"Medication '{medication_name}' not found in database records.",
    )


# X TODO: Step 2 - Construct your tools array including both the web search and your new custom roster tool
web_search = DuckDuckGoSearchRun()
tools = [web_search, lookup_specialist_department, get_clinic_stock_count]

# TODO: Step 3 - Initialize the autonomous agent executor with verbose output enabled
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are the AfyaPlus patient-support assistant.

            Use the available tools deliberately:
            - Use get_clinic_stock_count whenever the patient asks whether a medication is
                available, or asks about its quantity or pharmacy stock.
            - Use lookup_specialist_department when the patient asks which department or
                specialist handles a condition, or when the conversation indicates that a
                specialist referral is appropriate. Explain the referral or department
                information clearly, but do not diagnose or provide emergency medical
                advice.
            - Use web search only for general, current information that the local clinic
                tools cannot answer.

            Choose the smallest appropriate set of tools, and use the result of one tool
            to guide the next when the patient asks a compound question. Use chat_history
            as short-term memory: retain relevant details from earlier turns, such as the
            patient's medication, condition, and preferred context, and ask for
            clarification when they are missing or ambiguous. Never invent clinic stock,
            department details, or patient information.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent_session_history = {}
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def get_session_history(session_id: str):
    if session_id not in agent_session_history:
        agent_session_history[session_id] = ChatMessageHistory()
    return agent_session_history[session_id]


agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- Runtime Invocation Task ---
# Test the agent with two distinct steps using the same session to test prompt
# handling and memory management.
cfg = {"configurable": {"session_id": "patient-001"}}
agent_with_memory.invoke({"input": "My name is Achieng."}, cfg)  # type:ignore
agent_with_memory.invoke({"input": "Which clinic handles cardiology at AfyaPlus?"}, cfg)  # type:ignore
agent_with_memory.invoke(
    {
        "input": "Now find out what the general pre-booking prep is for cardiology on the web."
    },
    cfg,  # type:ignore
)
