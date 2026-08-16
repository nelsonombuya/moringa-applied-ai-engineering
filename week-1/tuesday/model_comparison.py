import time
from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
cloud_client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
local_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

SYSTEM_PROMPT = "You are a health assistant. Provide brief, safe guidance."
patient_message = "I have chest pain when I breathe deeply"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": patient_message},
]

# TODO 1: time the cloud call and store the response
start_time = time.time()
cloud_response = cloud_client.chat.completions.create(
    messages=messages,  # type: ignore
    model="gpt-4o-mini",
)
cloud_time = time.time() - start_time

# TODO 2: time the local call and store the response
start_time = time.time()
local_response = local_client.chat.completions.create(
    messages=messages,  # type: ignore
    model="llama3.2",
)
local_time = time.time() - start_time

# TODO 3: print a 3-row comparison: time, response length, first 200 chars
print("Time (seconds):")
print(f"Cloud: {cloud_time:.4f}")
print(f"Local: {local_time:.4f}")
print("\nResponse Length:")
print(f"Cloud: {len(cloud_response.choices[0].message.content)}")  # type: ignore
print(f"Local: {len(local_response.choices[0].message.content)}")  # type: ignore
print("\nFirst 200 Characters:")
print(f"Cloud: {cloud_response.choices[0].message.content[:200]}")  # type: ignore
print(f"Local: {local_response.choices[0].message.content[:200]}")  # type: ignore
