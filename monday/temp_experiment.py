import numpy as np


def predict_next_token(context, temperature=1.0):
    probs = {
        "and": 0.3,
        "with": 0.25,
        "recommend": 0.2,
        "suggesting": 0.15,
        "indicating": 0.1,
    }
    tokens = list(probs.keys())
    weights = np.array(list(probs.values()))
    scaled = weights ** (1.0 / temperature)
    scaled = scaled / scaled.sum()
    return np.random.choice(tokens, p=scaled)


temperatures = [0.1, 0.5, 1.0, 2.0]
num_trials = 10
# TODO: For each temperature, run num_trials predictions
# and count how often each token is selected.
for temp in temperatures:
    counts = {
        token: 0 for token in ["and", "with", "recommend", "suggesting", "indicating"]
    }
    for _ in range(num_trials):
        token = predict_next_token("patient has fever", temperature=temp)
        counts[token] += 1
    print(f"Temperature {temp}: {counts}")
