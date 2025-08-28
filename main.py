import argparse
import json
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# ----------------------------
# Utilities
# ----------------------------
def load_model(model_name: str) -> SentenceTransformer:
    print(f"Loading model: {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # Return L2-normalized embeddings for easy cosine similarity (dot product)
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # embeddings already normalized -> cosine = dot
    return A @ B.T

def top_k_search(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
    sims = (query_emb @ corpus_emb.T).reshape(-1)
    topk_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in topk_idx]

# ----------------------------
# Demos
# ----------------------------
def demo_similarity(model_name: str):
    model = load_model(model_name)
    pairs = [
        ("The cat sat on the mat.", "A cat is sitting on a rug."),
        ("I love pizza.", "The weather is rainy today."),
        ("How to apply for a driving license?", "Process to get a driver’s licence."),
        ("Python is a programming language.", "Snakes are reptiles.")
    ]
    texts = [t for p in pairs for t in p]
    embs = embed_texts(model, texts)
    print("\nPairwise cosine similarity:")
    for i, (a, b) in enumerate(pairs):
        s = float(embs[2*i] @ embs[2*i+1])
        print(f"- [{i+1}] \"{a}\"  <->  \"{b}\"  =>  {s:.3f}")

def demo_search(model_name: str, query: str, k: int):
    model = load_model(model_name)
    corpus = [
        "Renew my Indian passport in Hyderabad.",
        "Best pizza places near Gachibowli.",
        "How to integrate Azure Key Vault with .NET?",
        "RAG: retrieve and generate answers using embeddings.",
        "TDS on property purchase in India.",
        "Configuring Service Bus sessions for ordered processing.",
        "Weekend trek spots near Bangalore."
    ]
    q_emb = embed_texts(model, [query])[0]
    c_emb = embed_texts(model, corpus)
    hits = top_k_search(q_emb, c_emb, k)
    print(f'\nQuery: "{query}"')
    print("Top matches:")
    for rank, (idx, score) in enumerate(hits, 1):
        print(f"{rank:2d}. ({score:.3f}) {corpus[idx]}")

def demo_cluster(model_name: str, k: int):
    model = load_model(model_name)
    corpus = [
        # Tech
        "How to scale Azure Functions with Service Bus?",
        "Configuring EF Core connection pooling.",
        "Best practices for SQL Server indexing.",
        "Building a FastAPI service with uvicorn.",
        # Food
        "Hyderabadi biryani vs Lucknowi biryani.",
        "Top dosa places near HSR Layout.",
        "Perfect pizza dough hydration percentage.",
        # Travel
        "One-day trip plan for Ramoji Film City.",
        "Weekend getaway from Bangalore to Coorg.",
        "Must-see places around Charminar."
    ]
    embs = embed_texts(model, corpus)
    # KMeans on normalized embeddings
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(embs)

    print("\nClusters:")
    for ci in range(k):
        print(f"\n== Cluster {ci} ==")
        for i, s in enumerate(corpus):
            if labels[i] == ci:
                print("-", s)

def demo_knn_classify(model_name: str):
    """
    Tiny zero-config text classifier using k-NN on embeddings.
    Labels: TECH, FOOD, TRAVEL
    """
    model = load_model(model_name)

    train = [
        ("Configuring Azure Service Bus topics", "TECH"),
        ("SQL indexing strategies for large tables", "TECH"),
        ("Angular vs React for enterprise apps", "TECH"),
        ("Best biryani in Hyderabad", "FOOD"),
        ("Dosa batter fermentation tips", "FOOD"),
        ("Neapolitan pizza at home", "FOOD"),
        ("Day trip to Ramoji Film City", "TRAVEL"),
        ("Weekend trek near Bangalore", "TRAVEL"),
        ("Places to visit around Charminar", "TRAVEL"),
    ]

    X_train = embed_texts(model, [t for t, _ in train])
    y_train = np.array([l for _, l in train])

    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    # scikit-learn's cosine needs raw vectors; we give normalized ones and use 1 - cosine via metric param.
    # But KNeighborsClassifier doesn't accept 'cosine' directly in all versions; fallback using brute force:
    # We'll approximate by using Euclidean on normalized vectors (works because ||u-v||^2 = 2(1 - cos))
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(X_train, y_train)

    tests = [
        "Integrate Key Vault with .NET",
        "Where to eat the best dosa in Bangalore?",
        "Quick road trip from Hyderabad for a day"
    ]
    X_test = embed_texts(model, tests)
    preds = knn.predict(X_test)

    print("\nPredictions:")
    for t, p in zip(tests, preds):
        print(f"- {p:7s} :: {t}")

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="GTE-small NLP playground")
    parser.add_argument("--model", default="thenlper/gte-small",
                        help="Hugging Face model id (e.g., thenlper/gte-small, BAAI/bge-small-en)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("similarity", help="Pairwise sentence similarity demo")
    search = sub.add_parser("search", help="Semantic search over a tiny corpus")
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=3)

    cluster = sub.add_parser("cluster", help="KMeans clustering over sample corpus")
    cluster.add_argument("--k", type=int, default=3)

    knn = sub.add_parser("classify", help="k-NN text classifier demo")

    args = parser.parse_args()

    if args.cmd == "similarity":
        demo_similarity(args.model)
    elif args.cmd == "search":
        demo_search(args.model, args.query, args.k)
    elif args.cmd == "cluster":
        demo_cluster(args.model, args.k)
    elif args.cmd == "classify":
        demo_knn_classify(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()