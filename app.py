modelName = "thenlper/gte-small"
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

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


def fetchExpenseCategoryUsingEmbeddings(expense_text: str):
    model = SentenceTransformer(modelName, trust_remote_code=True)
    # Embed expense
    expense_emb = model.encode([expense_text], normalize_embeddings=True)
    label_embeddings = model.encode(labels, normalize_embeddings=True)

    # Cosine similarity = dot product (since normalized)
    scores = np.dot(expense_emb, label_embeddings.T)[0]

    # Pick highest-scoring label
    best_idx = np.argmax(scores)
    return labels[best_idx], scores


def fetchExpenseCategoryUsingBert(expense_text: str):
    model_id = "MoritzLaurer/deberta-v3-base-mnli"
    classifier = pipeline("zero-shot-classification", model=model_id)
    result = classifier(expense_text, candidate_labels=labels)

    return result["labels"][0]


print("Label using embeddings")
category = fetchExpenseCategoryUsingEmbeddings("Loan EMI")
print(category[0])
print("Label using Bert")
category = fetchExpenseCategoryUsingBert("Loan EMI")
print(category)
