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
    "LANGUAGE RULES: Detect the language of the patient message. "
    "ALWAYS respond in that same language. "
    "Supported languages: English, Swahili, Sheng.\n\n"
    "You are an AfyaPlus health assistant. Provide brief, safe guidance. "
    "Never diagnose or prescribe."
)
test_messages = [
    "I have a fever and headache for two days",
    "Nina maumivu ya kichwa kwa siku tatu",  # Swahili: I have a headache for three days
    "Niko na homa na maumivu ya tumbo",  # Swahili: I have a fever and stomach pain
    "Niko na kichwa na unchungu kwa koo",  # Sheng: I have a cough and sore throat
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
