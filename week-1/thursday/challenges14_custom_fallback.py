# python: challenge14_custom_fallback.py
# Reuse imports and validate_response from Lab 14.
import time
from os import environ

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from .production_pipeline import FALLBACK_RESPONSE, validate_response

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# TODO 1 (done): Define FALLBACK_TIMEOUT, FALLBACK_RATE_LIMIT, FALLBACK_API_ERROR
#         as three distinct strings.
FALLBACK_TIMEOUT = "I am currently unable to process your request due to a timeout. Please try again later."
FALLBACK_RATE_LIMIT = "I am currently unable to process your request due to high demand. Please try again later."
FALLBACK_API_ERROR = "I am currently unable to process your request due to a server error. Please try again later."


def get_ai_response(patient_message, max_retries=3):
    last_error = None  # Track the last exception type seen.
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are AfyaPlus Health Assistant. Provide safe, general health guidance. Never diagnose or prescribe.",
                    },
                    {"role": "user", "content": patient_message},
                ],
                temperature=0.3,
                max_tokens=200,
                timeout=10.0,
            )
            ai_text = response.choices[0].message.content
            is_safe, reason = validate_response(ai_text)
            if not is_safe:
                print(f"  [SAFETY] Response blocked: {reason}")
                # Use a generic fallback for blocked responses.
                return FALLBACK_RESPONSE
            return ai_text
        except APITimeoutError:
            last_error = "timeout"
            time.sleep(2**attempt)
        except RateLimitError:
            last_error = "rate_limit"
            time.sleep(5 * (attempt + 1))
        except APIError:
            last_error = "api_error"
            # Small backoff added so repeated generic API errors don't hammer
            # the endpoint with zero delay between retries.
            time.sleep(1)

    # TODO 2 (done): Branch on last_error and return the matching fallback.
    if last_error == "timeout":
        return FALLBACK_TIMEOUT
    if last_error == "rate_limit":
        return FALLBACK_RATE_LIMIT
    if last_error == "api_error":
        return FALLBACK_API_ERROR
    # Defensive default: should not normally be reached, but avoids returning
    # an undefined value if the loop exits without recording an error type.
    return FALLBACK_RESPONSE
