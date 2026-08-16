import os
import sys

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# TODO 1: Exit with sys.exit(1) and a friendly message if api_key is None.
if api_key is None:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    sys.exit(1)
# TODO 2: Exit if api_key equals the placeholder 'sk-your-key-here'.
if api_key == "sk-your-key-here":
    print("ERROR: Please replace the placeholder API key with your actual OPENAI_API_KEY")
    sys.exit(1)
# TODO 3: Otherwise print 'Key loaded successfully' with the masked key.
print("Key loaded successfully")
print(f"Masked API Key: {api_key[:5]}...")
