from datetime import UTC, datetime
from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


class ConversationLogger:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.start_time = datetime.now(tz=UTC).isoformat()
        self.messages = []
        self.total_tokens = 0
        self.turns = 0

    def add_exchange(self, user_text, assistant_text, tokens_used):
        # TODO 1: Append the user and assistant messages with timestamps.
        self.messages.append(
            {
                "user": user_text,
                "assistant": assistant_text,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
        )
        # TODO 2: Increment self.turns and self.total_tokens.
        self.turns += 1
        self.total_tokens += tokens_used

    def save_log(self, path):
        # TODO 3: Write {patient_id, start_time, end_time, turns, total_tokens, messages}
        #         to 'path' as pretty JSON.
        end_time = datetime.now(tz=UTC).isoformat()
        log_data = {
            "patient_id": self.patient_id,
            "start_time": self.start_time,
            "end_time": end_time,
            "turns": self.turns,
            "total_tokens": self.total_tokens,
            "messages": self.messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(log_data, f, indent=4, ensure_ascii=False)


# Test out the conversation logger with a sample conversation.
if __name__ == "__main__":
    logger = ConversationLogger(patient_id="test_patient")
    logger.add_exchange(
        user_text="Hello, I have a headache.",
        assistant_text="I'm sorry to hear that. Can you tell me more about your symptoms?",
        tokens_used=15,
    )
    logger.add_exchange(
        user_text="It's been going on for two days.",
        assistant_text="Have you taken any medication for it?",
        tokens_used=12,
    )
    logger.save_log("test_conversation_log.json")
