import numpy as np
import faiss
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer


# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reranker: HuggingFace Cross-Encoder (Fast)
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)

# Embedder: Sentence-Transformer MiniLM
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_index_and_metadata(index_path="vector_store/index.faiss", meta_path="vector_store/metadata.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve_top_k(query: str, k=10):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    index, metadata = load_index_and_metadata()
    distances, indices = index.search(query_embedding, k)

    retrieved = []
    for rank, i in enumerate(indices[0]):
        meta = metadata[i]
        retrieved.append({
            "text": meta["text"],
            "page": meta["page"],
            "chunk_index": meta["chunk_index"],
            "score": float(distances[0][rank])
        })

    return retrieved


def rerank_chunks(query: str, chunks: list, top_n=5):
    """
    Rerank FAISS-retrieved chunks using Cross-Encoder.

    Args:
        query: User input string
        chunks: List of chunks (with "text")
        top_n: Return top-N reranked

    Returns:
        List of top-N chunks sorted by relevance
    """
    input_pairs = [(query, chunk["text"]) for chunk in chunks]
    inputs = reranker_tokenizer.batch_encode_plus(
        input_pairs, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()

    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# query = "Is icu treatment covered under this policy and are there any limits?"
# top_chunks = retrieve_top_k(query, k=10)
# final_chunks = rerank_chunks(query, top_chunks, top_n=5)

# for c in final_chunks:
#     print(f"\n[Page {c['page']}] Score: {c['rerank_score']:.4f}")
#     print(c["text"][:300], "...")
