# Same imports/client/function as Lab 8.
# The only change is inside messages[0]['content']:
from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = (
    "You are an expert emergency triage nurse at AfyaPlus Health. "
    "Analyse the user symptoms. You MUST explain your clinical "
    "reasoning step-by-step BEFORE concluding with a final directive. "
    "Follow this EXACT structural layout:\n\n"
    "REASONING STEPS:\n- [Step 1]\n- [Step 2]\n"
    "FINAL DIRECTIVE: [Emergency Room / Clinic Appointment / Home Care]\n"
    # TODO 1: Append an additional line that requires CONFIDENCE: [HIGH/MEDIUM/LOW].
    "CONFIDENCE: [HIGH/MEDIUM/LOW]\n"
    # TODO 2: Add a sentence telling the model to use LOW whenever reasoning steps
    "If any critical information is missing, set CONFIDENCE to LOW.\n"
    #         indicate missing information.
)


# Test with a clear case and a vague case to confirm CONFIDENCE adapts.
def run_triage_reasoning(symptom_report):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": symptom_report},
        ],
    )
    return response.choices[0].message.content


print(
    run_triage_reasoning(
        "I bumped my head an hour ago. "
        "I felt fine at first, but now I am getting dizzy and nauseous."
    )
)
print(run_triage_reasoning("I have a headache."))
