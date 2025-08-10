import os
import json
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Embedding model for retrieval
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_index_and_metadata(index_path, meta_path):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata file not found.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve_top_k(query: str, k: int, index_path: str, meta_path: str) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks based on semantic similarity.
    
    Args:
        query: Query string
        k: Number of top chunks to retrieve (default changed to 10)
        index_path: Path to FAISS index
        meta_path: Path to metadata JSON
        
    Returns:
        List of top-k chunks with similarity scores
    """
    query_emb = embed_model.encode(["query: " + query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    index, metadata = load_index_and_metadata(index_path, meta_path)
    distances, indices = index.search(query_emb, k)
    results = []
    for rank, i in enumerate(indices[0]):
        if i >= len(metadata):
            continue
        meta = metadata[i]
        results.append({
            "text": meta.get("text", ""),
            "page": meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", i),
            "score": float(distances[0][rank])
        })
    return results


# Example usage:
# query = "Is ICU treatment covered under this policy and are there any limits?"
# top_chunks = retrieve_top_k(query, k=10, index_path="...", meta_path="...")

# for chunk in top_chunks:
#     print(f"\n[Page {chunk['page']}] Similarity Score: {chunk['score']:.4f}")
#     print(chunk['text'][:300] + "...")
