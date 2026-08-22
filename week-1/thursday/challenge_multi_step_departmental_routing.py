import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize our engineering infrastructure backends
load_dotenv()
llm_router = ChatOpenAI(
    temperature=0.0,
    model="openai/gpt-4o-mini",
    max_tokens=5,  # type:ignore
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  # type:ignore
)
llm_chain = ChatOpenAI(
    temperature=0.2,
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  # type:ignore
)
# --- TASK 1: DEFINE YOUR COMPONENTS ---
# TODO: Define your classification routing prompt template
router_prompt = ChatPromptTemplate.from_template(
    "Classify the patient message as exactly one word, either 'CLINICAL' or 'BILLING'.\n"
    "Message: {message}\nClassification:"
)

# TODO: Define your step 1 generation prompts for both domains
clinical_template = ChatPromptTemplate.from_template(
    "You are a medical triage assistant. "
    "Draft a 1-sentence medical triage acknowledgement for this message: {message}"
)
billing_template = ChatPromptTemplate.from_template(
    "You are a billing support assistant. "
    "Draft a 1-sentence billing support acknowledgement for this message: {message}"
)

# TODO: Define your step 2 Swahili translation template
translation_template = ChatPromptTemplate.from_template(
    "Translate this text into clear Swahili for an SMS alert: {text}"
)

# --- TASK 2: BUILD LCEL PIPELINES ---
# TODO: Construct your router chain and your translation chain links
router_chain = router_prompt | llm_router | StrOutputParser()
billing_chain = billing_template | llm_chain | StrOutputParser()
clinical_chain = clinical_template | llm_chain | StrOutputParser()
translation_chain = translation_template | llm_chain | StrOutputParser()

# --- TASK 3: CONSTRUCT RUNTIME FLOW ---
incoming_patient_sms = ChatPromptTemplate.from_template(
    "I need to request an itemized receipt for my outpatient prescription payment."
)
print(f"Incoming Payload: {incoming_patient_sms}\n")

# TODO: Step A - Run the router chain to evaluate the target category
detected_route = router_chain.invoke({"message": incoming_patient_sms})
print(f"--- Route Evaluated: {detected_route} ---")

# TODO: Step B - Execute the corresponding multi-step chain execution paths based on the route
# Clean the string, run step 1, pass the result to step 2, and print the final Swahili output
if detected_route == "CLINICAL":
    step1_output = clinical_chain.invoke({"message": incoming_patient_sms})
    step2_output = translation_chain.invoke({"text": step1_output})
    print(f"Final Swahili SMS Wire: {step2_output}")
elif detected_route == "BILLING":
    step1_output = billing_chain.invoke({"message": incoming_patient_sms})
    step2_output = translation_chain.invoke({"text": step1_output})
    print(f"Final Swahili SMS Wire: {step2_output}")
else:
    print(f"Error: Unrecognized route '{detected_route}'")
