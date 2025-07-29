
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


# 1. Load BGE model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def generate_embeddings(texts: list[str]) -> np.ndarray:
    return np.array(model.encode(
        texts,
        normalize_embeddings=True,  # Important for cosine similarity
        show_progress_bar=True
    ))

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
    index.add(embeddings)
    return index

def save_index(index, metadata, out_path="faiss_index"):
    faiss.write_index(index, f"{out_path}.index")
    with open(f"{out_path}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

# === MAIN FLOW ===
chunks = preprocess_pdf("/content/dataset1.pdf")
texts = [chunk["text"] for chunk in chunks]

# ✅ Include full text in metadata for each chunk
metadata = [
    {
        **chunk["meta"],
        "text": chunk["text"]
    }
    for chunk in chunks
]

embeddings = generate_embeddings(texts)
index = build_faiss_index(embeddings)
save_index(index, metadata)
