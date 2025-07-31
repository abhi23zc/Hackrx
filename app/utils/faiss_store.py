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


# from embedder import generate_embeddings
# from pdf_extractor import extract_pdf_content
# from chunker import chunk_text
# from faiss_store import create_faiss_index, save_metadata, search, load_metadata

# # Step 1: Load & Process
# pdf_data = extract_pdf_content("dataset1.pdf")
# chunks = chunk_text(pdf_data["pages"])
# texts, embeddings = generate_embeddings(chunks)

# # Step 2: Store
# create_faiss_index(embeddings)
# save_metadata(chunks)

# # Step 3: Query
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# query = "Summarize the report findings"
# query_emb = model.encode([query])
# indices, scores = search(np.array(query_emb).astype("float32"))

# # Step 4: Retrieve
# metadata = load_metadata()
# top_chunks = [metadata[i] for i in indices]

# print("Top Chunks:")
# for chunk in top_chunks:
#     print(f"Page {chunk['page']}: {chunk['text'][:200]}...\n")

