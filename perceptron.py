import numpy as np
import random

def train(positive_examples, negative_examples, num_iterations=100, eta=1):
    weights = np.array([0.0, 0.0, 0.0])  # Initialize weights
        
    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)
        if z < 0:  # positive example misclassified
            weights = weights + eta * pos  # ← bug fixed

        z = np.dot(neg, weights)
        if z >= 0:  # negative example misclassified
            weights = weights - eta * neg  # ← bug fixed

    return weights


# ── DATA ──────────────────────────────────────────────────────
# Each point = [tumor_size, patient_age, 1]
#                                        ↑ bias term (always 1)

positive_examples = [   # Malignant (cancerous)
    np.array([20, 60, 1]),
    np.array([18, 55, 1]),
    np.array([22, 70, 1]),
    np.array([25, 65, 1]),
]

negative_examples = [   # Benign (harmless)
    np.array([2, 25, 1]),
    np.array([3, 30, 1]),
    np.array([1, 20, 1]),
    np.array([4, 28, 1]),
]

# ── TRAIN ─────────────────────────────────────────────────────
weights = train(positive_examples, negative_examples, num_iterations=100, eta=1)
print(f"Learned weights: {weights}")


# ── PREDICT ───────────────────────────────────────────────────
def predict(point, weights):
    z = np.dot(point, weights)
    if z >= 0:
        return "Malignant (Positive)"
    else:
        return "Benign (Negative)"


# ── TEST ──────────────────────────────────────────────────────
test_cases = [
    np.array([21, 63, 1]),   # large tumor, old  → expect Malignant
    np.array([2,  22, 1]),   # small tumor, young → expect Benign
    np.array([19, 58, 1]),   # large tumor, old  → expect Malignant
    np.array([3,  27, 1]),   # small tumor, young → expect Benign
]

print("\n--- Predictions ---")
for test in test_cases:
    result = predict(test, weights)
    print(f"  Tumor size={test[0]}, Age={test[1]}  →  {result}")


