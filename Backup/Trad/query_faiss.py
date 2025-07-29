
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# === Config ===
MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_PATH = "faiss_index.index"
META_PATH = "faiss_index_meta.pkl"
TOP_K = 5

# === Load model, index, metadata ===
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

def search(query: str, top_k=TOP_K):
    formatted_query = "query: " + query
    query_embedding = model.encode(formatted_query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype("float32")

    scores, indices = index.search(query_embedding, top_k)
    results = []

    for i, idx in enumerate(indices[0]):
        result = {
            "rank": i + 1,
            "score": float(scores[0][i]),
            "meta": {
                "page": metadata[idx].get("page"),
                "section": metadata[idx].get("section", "Unknown"),
                "text": metadata[idx].get("text", "[NO TEXT FOUND]")  # ✅ Ensure text is included
            }
        }
        results.append(result)

    return results


# === Function: Search from query ===
# def search(query: str, top_k=TOP_K):
#     # BGE models benefit from "query: ..." format
#     formatted_query = "query: " + query
#     query_embedding = model.encode(formatted_query, normalize_embeddings=True)
#     query_embedding = np.array([query_embedding]).astype("float32")

#     # FAISS similarity search
#     scores, indices = index.search(query_embedding, top_k)
#     results = []

#     for i, idx in enumerate(indices[0]):
#         result = {
#             "rank": i + 1,
#             "score": float(scores[0][i]),
#             "meta": metadata[idx]
#         }
#         results.append(result)

#     return results

if __name__ == "__main__":
    user_query = "Will any policy cover attempt at suicide"

    top_chunks = search(user_query, top_k=5)

    print("\nTop Relevant Chunks:\n")
    for result in top_chunks:
        rank = result['rank']
        score = result['score']
        meta = result['meta']

        print(f"[{rank}] Score: {score:.4f}")
        print(f"Page: {meta.get('page')} | Section: {meta.get('section')}")

        # ✅ Print a preview of the text to confirm it's valid
        chunk_text = meta.get('text', '[NO TEXT FOUND]')
        print("Text Preview:", chunk_text[:1000].strip())  # Print first 300 chars
        print("-" * 100)