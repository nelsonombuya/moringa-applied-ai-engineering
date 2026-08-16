import numpy as np


def simple_numpy_tokenizer(text):
    clean_text = text.lower().replace(".", "").replace("?", "").replace(",", "")
    tokens = clean_text.split()
    return np.array(tokens)


original_query = (
    "Hujambo, it is Juma. I am feeling very hot, my head hurts since yesterday, "
    "and I am coughing. I cannot go to the clinic because of the rain. What should I do?"
)

# TODO 1: Tokenise the original_query and count the tokens.
original_tokens = simple_numpy_tokenizer(original_query)
original_token_count = len(original_tokens)

# TODO 2: Manually rewrite the query as a 10-word summary of the medical facts only.
summary = (
    "Juma is feeling hot, has a headache, and is coughing. He can't go to the "
    "clinic due to rain."
)

# TODO 3: Tokenise the summary and count the tokens.
summary_tokens = simple_numpy_tokenizer(summary)
summary_token_count = len(summary_tokens)

# TODO 4: Calculate the percentage of tokens saved.
tokens_saved = original_token_count - summary_token_count
percentage_saved = (tokens_saved / original_token_count) * 100

# TODO 5: If 1,000 users send this query daily, how many tokens are saved per month?
monthly_tokens_saved = tokens_saved * 1000 * 30

print(f"Original token count: {original_token_count}")
print(f"Summary token count: {summary_token_count}")
print(f"Tokens saved: {tokens_saved}")
print(f"Percentage of tokens saved: {percentage_saved:.2f}%")
print(f"Monthly tokens saved (for 1,000 users): {monthly_tokens_saved}")
