# Reuse the imports and stream setup from streaming_response.py.
from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

patient_message = "I have been feeling very tired for the past week and have no appetite."

print(f"Patient: {patient_message}")
print("Assistant: ", end="", flush=True)

try:
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are the AfyaPlus Health Assistant. Provide brief, "
                "empathetic guidance.",
            },
            {"role": "user", "content": patient_message},
        ],
        temperature=0.3,
        max_tokens=200,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token

    print()
    print("--- Streaming complete ---")
    print(f"Total response length: {len(full_response)} characters")

    last_milestone = 0
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        print(delta, end="", flush=True)
        full_response += delta
        # TODO 1: Compute current_word_count = len(full_response.split()).
        current_word_count = len(full_response.split())
        # TODO 2: When current_word_count crosses a new multiple of 10,
        if current_word_count % 10 == 0 and current_word_count != last_milestone:
            print(f" [{current_word_count} words]", end="", flush=True)
            last_milestone = current_word_count
        # TODO 3: Use last_milestone to avoid printing the same milestone twice.


except Exception as e:
    print(f"\nError during streaming: {e}")
