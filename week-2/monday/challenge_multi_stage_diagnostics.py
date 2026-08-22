import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.3,
    api_key=os.environ["OPENROUTER_API_KEY"],  # type:ignore
    base_url="https://openrouter.ai/api/v1",
)

prompt_explanation = ChatPromptTemplate.from_template(
    "You are a medical expert. "
    "Explain the following medical condition in simple terms for a patient: {condition}"
)
prompt_precaution = ChatPromptTemplate.from_template(
    "You are a medical expert. "
    "What are the key precautions a patient should take for {condition}?"
)
prompt_translation = ChatPromptTemplate.from_template(
    "You are a translation assistant. Translate the following text into Swahili: {text}"
)

# --- LCEL Engine Piping ---
# TODO: Assemble your distinct chains using the pipe operator (|) and StrOutputParser()
chain_explanation = prompt_explanation | llm | StrOutputParser()
chain_precaution = prompt_precaution | llm | StrOutputParser()
chain_translation = prompt_translation | llm | StrOutputParser()

# --- Execution Runtime ---
target_condition = "High fever accompanied by severe chills (Suspected Malaria)"
print(f"Target Condition: {target_condition}\n")

# TODO: Invoke Chain 1 to get the explanation
explanation_output = chain_explanation.invoke({"condition": target_condition})

# TODO: Invoke Chain 2 by passing the output of Chain 1
precaution_output = chain_precaution.invoke({"condition": target_condition})

# TODO: Invoke Chain 3 by passing both upstream string outputs simultaneously
final_swahili_sms = chain_translation.invoke(
    {"text": f"{explanation_output} {precaution_output}"}
)

# Print output parameters
print(f"Generated Explanation: {explanation_output}\n")
print(f"Generated Precaution: {precaution_output}\n")
print(f"Final Swahili SMS Wire: {final_swahili_sms}")
