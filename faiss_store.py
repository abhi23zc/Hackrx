import faiss
import numpy as np
import json
import os

def create_faiss_index(embeddings, index_path="vector_store/index.faiss", use_cosine=True):
    """
    Create and save a FAISS index from embeddings.
    
    Args:
        embeddings: NumPy array of shape (N, D)
        index_path: Path to save the FAISS index
        use_cosine: If True, normalize and use cosine similarity
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    dim = embeddings.shape[1]

    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)  # cosine = inner product after normalization
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")


def save_metadata(chunks, meta_path="vector_store/metadata.json"):
    """
    Save chunk metadata (e.g., page, chunk index, text).
    """
    metadata = [
        {
            "text": chunk["text"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


def load_faiss_index(index_path="vector_store/index.faiss"):
    return faiss.read_index(index_path)


def load_metadata(meta_path="vector_store/metadata.json"):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query_embedding, k=5, use_cosine=True, index_path="vector_store/index.faiss"):
    """
    Perform vector similarity search on FAISS index.

    Returns:
        List of top-k indices and scores.
    """
    index = load_faiss_index(index_path)
    
    if use_cosine:
        faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]
