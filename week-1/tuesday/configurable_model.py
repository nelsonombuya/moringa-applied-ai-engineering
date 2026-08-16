from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# TODO 1: Load MODEL_BASE_URL from env with a sensible default.
model_base_url = getenv("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
# TODO 2: Load MODEL_NAME from env with a sensible default.
model_name = getenv("MODEL_NAME", "gpt-4o-mini")
# TODO 3: Initialise OpenAI(base_url=..., api_key=...) using the loaded values.
client = OpenAI(base_url=model_base_url, api_key=getenv("OPENROUTER_API_KEY"))
# TODO 4: Pass MODEL_NAME into client.chat.completions.create(model=...).


client = OpenAI(
    base_url=model_base_url,
    api_key=getenv("OPENROUTER_API_KEY"),
)
model = getenv("MODEL_NAME", "gpt-4o-mini")

response = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Say hello"}]
)
print(response.choices[0].message.content)
