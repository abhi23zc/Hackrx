import os
import json
import torch
import numpy as np
import faiss
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ---- Reranker Model: BGE Reranker Large ----
reranker_model_name = "BAAI/bge-reranker-base"

reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)


def load_index_and_metadata(index_path: str = "vector_store/index.faiss",
                            meta_path: str = "vector_store/metadata.json"):
    """
    Load FAISS index and associated metadata.

    Returns:
        FAISS index and metadata list
    """
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata file not found.")
    
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata


def retrieve_top_k(query: str, k: int = 10) -> List[Dict]:
    """
    Retrieve top-K similar chunks from FAISS index using dense embedding.

    Args:
        query: Input user query
        k: Number of top results to retrieve

    Returns:
        List of chunk metadata with similarity scores
    """
    query_emb = embed_model.encode(["query: " + query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    index, metadata = load_index_and_metadata()
    distances, indices = index.search(query_emb, k)

    results = []
    for rank, i in enumerate(indices[0]):
        if i >= len(metadata):
            continue  # Avoid index errors
        meta = metadata[i]
        results.append({
            "text": meta.get("text", ""),
            "page": meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", i),
            "score": float(distances[0][rank])
        })

    return results


def rerank_chunks(query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Rerank retrieved chunks using a Cross-Encoder (reranker model).

    Args:
        query: Original query string
        chunks: Retrieved chunks with text
        top_n: How many top reranked results to return

    Returns:
        Top-N reranked chunks
    """
    input_pairs = [(query, chunk["text"]) for chunk in chunks]
    inputs = reranker_tokenizer.batch_encode_plus(
        input_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()

    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# Example usage:
# query = "Is ICU treatment covered under this policy and are there any limits?"
# top_chunks = retrieve_top_k(query, k=10)
# final_chunks = rerank_chunks(query, top_chunks, top_n=5)

# for chunk in final_chunks:
#     print(f"\n[Page {chunk['page']}] Rerank Score: {chunk['rerank_score']:.4f}")
#     print(chunk['text'][:300] + "...")
