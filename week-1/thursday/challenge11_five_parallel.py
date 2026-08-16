# Reuse the imports and async functions from async_calls.py.
from asyncio import gather, run

from async_calls import process_patient

patient_messages = [
    "I have a persistent cough for two weeks",
    "My child has a rash on their arms",
    "I feel dizzy when I stand up quickly",
    # TODO 1: Add a fourth realistic patient message.
    "I have been experiencing severe headaches for the past three days",
    # TODO 2: Add a fifth realistic patient message.
    "I have noticed swelling in my ankles and feet recently",
]


# The rest of the script is identical: asyncio.gather scales automatically.
async def main():
    tasks = [process_patient(msg, i) for i, msg in enumerate(patient_messages, 1)]
    results = await gather(*tasks)
    for result in results:
        print(result)
        print()


run(main())
