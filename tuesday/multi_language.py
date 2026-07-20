from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


# TODO: Replace this placeholder with a system prompt that instructs
# the model to detect the patient's language and respond in the same language.
SYSTEM_PROMPT = (
    "You are an AfyaPlus health assistant. Provide brief guidance."
    " Detect the patient's language and respond in the same language. "
    "Never diagnose conditions or prescribe medication. "
    "Keep your responses concise and safe. "
)
test_messages = [
    "I have a fever and headache for two days",
    "Nina maumivu ya kichwa kwa siku tatu",  # Swahili: I have a headache for three days
]

for msg in test_messages:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    print(f"Patient: {msg}")
    print(f"Assistant: {response.choices[0].message.content}\n")
