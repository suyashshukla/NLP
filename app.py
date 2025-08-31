modelName = "thenlper/gte-small"
from ast import List
import numpy as np
from sentence_transformers import SentenceTransformer

labels = [
    "Food",
    "Travel",
    "Shopping",
    "Entertainment",
    "Bills",
    "Salary",
    "Grocery",
    "Gadgets",
    "Dairy",
]


def fetchCategory(expense_text: str):
    model = SentenceTransformer(modelName, trust_remote_code=True)
    # Embed expense
    expense_emb = model.encode([expense_text], normalize_embeddings=True)
    label_embeddings = model.encode(labels, normalize_embeddings=True)

    # Cosine similarity = dot product (since normalized)
    scores = np.dot(expense_emb, label_embeddings.T)[0]

    # Pick highest-scoring label
    best_idx = np.argmax(scores)
    return labels[best_idx], scores


category, scores = fetchCategory("Milk, Bread and Butter")
print(f"Category: {category}")
print("Scores:")
for label, score in zip(labels, scores):
    print(f" - {label}: {score:.4f}")
