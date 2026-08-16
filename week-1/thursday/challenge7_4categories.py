from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


def classify(new_patient_query):
    """
    Classify incoming user medical queries into exactly one category: CRITICAL,
    NON_URGENT, ROUTINE, or EMERGENCY_DISPATCH.
    The system message defines the task; the user/assistant pairs demonstrate the EXACT
    output format; the final user message is the live target the model will classify in
    the same style.

    Args:
        new_patient_query (str): The medical query to classify.
    Returns:
        str: The model's classification of the query.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            # TODO 1: Update the system message to list all FOUR categories
            # ---- ROLE & TASK ----
            {
                "role": "system",
                "content": "Classify incoming user medical queries into exactly "
                "one category: CRITICAL, NON_URGENT, ROUTINE, or EMERGENCY_DISPATCH.",
            },
            # ---- FEW-SHOT EXAMPLE 1: a CRITICAL case ----
            {
                "role": "user",
                "content": "Query: I cannot breathe and my left arm feels numb.",
            },
            {"role": "assistant", "content": "Category: CRITICAL"},
            # ---- FEW-SHOT EXAMPLE 2: a ROUTINE case ----
            {
                "role": "user",
                "content": "Query: I need to renew my allergy pills prescription "
                "next month.",
            },
            {"role": "assistant", "content": "Category: ROUTINE"},
            # TODO 2: Add one new EMERGENCY_DISPATCH example pair here.
            # --- FEW-SHOT EXAMPLE 3: an EMERGENCY_DISPATCH case ----
            {
                "role": "user",
                "content": "Query: I've been shot and I haven't stopped bleeding.",
            },
            {"role": "assistant", "content": "Category: EMERGENCY_DISPATCH"},
            # ---- LIVE TARGET QUERY ----
            {"role": "user", "content": f"Query: {new_patient_query}"},
        ],
    )
    return response.choices[0].message.content


print(classify("I have a small bruise on my knee"))  # NON_URGENT
print(
    classify("Severe bleeding that will not stop after 20 minutes")
)  # EMERGENCY_DISPATCH
