from os import environ

from dotenv import load_dotenv
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.schema import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()


def route_patient_query(message: str):
    # TODO: Replace this facility-name check with a zero-temperature LLM
    # classifier (reuse Lab 4's router_chain pattern) that labels `message`
    # as INFO or EMERGENCY. If EMERGENCY, return a fixed, hardcoded response
    # right here — this branch must never reach the agent below.
    # A low temperature (0.0) removes creative variation, making classification deterministic.
    router_llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=5,  # type:ignore
        api_key=environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    router_prompt = ChatPromptTemplate.from_template(
        "Classify the patient message as exactly one word, either 'INFO' or 'EMERGENCY'.\n"
        "Message: {message}\nClassification:"
    )

    router_chain = router_prompt | router_llm | StrOutputParser()
    classification = router_chain.invoke({"message": message}).upper().strip()
    print(f"Routed to: {classification}")

    if classification == "EMERGENCY":
        return "[EMERGENCY ROUTE] Please call 911 or go to the nearest emergency room immediately."

    # Dynamic Agent execution path — used only for INFO-classified messages
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        api_key=environ.get("OPENROUTER_API_KEY"),  # type:ignore
        base_url="https://openrouter.ai/api/v1",
    )
    search_tool = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Regional_Lookup",
            func=search_tool.run,
            description="Finds current clinics.",
        )
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent_response = agent.invoke({"input": message})
    return f"[DYNAMIC AGENT ROUTE] Extracted Online: {agent_response['output']}"


# X TODO: Test the router with a sample patient query
if __name__ == "__main__":
    # X TODO: Test an INFO Request
    test_message = "Where can I find a pediatric clinic nearby?"
    response = route_patient_query(test_message)
    print(response)

    # X TODO: Test an EMERGENCY Request
    test_message = "I am having severe chest pain and difficulty breathing."
    response = route_patient_query(test_message)
    print(response)
