# import torch
# import numpy as np
# from InstructorEmbedding import INSTRUCTOR

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load Instructor-XL (once)
# model = INSTRUCTOR("hkunlp/instructor-xl")
# model.to(DEVICE)

# def generate_embeddings(chunks, batch_size=16):
#     """
#     Generate embeddings using Instructor-XL for RAG.
    
#     Args:
#         chunks: List of dicts with 'text' (from chunker)
#         batch_size: Number of chunks processed in one batch
    
#     Returns:
#         Tuple of:
#             - List of (instruction, chunk) inputs
#             - NumPy array of embeddings (shape: [N, D])
#     """
#     # Format inputs: [ [instruction, text], ... ]
#     input_pairs = [["Represent the document for retrieval", chunk["text"]] for chunk in chunks]
#     embeddings = []

#     for i in range(0, len(input_pairs), batch_size):
#         batch = input_pairs[i:i + batch_size]
#         with torch.no_grad():
#             emb = model.encode(batch, convert_to_tensor=False)
#         embeddings.extend(emb)

#     return input_pairs, np.array(embeddings)

# from chunker import chunk_text
# from pdf_extractor import extract_pdf_content
# from embedder import generate_embeddings

# # Step 1: Extract + Chunk
# pdf_data = extract_pdf_content("dataset1.pdf")
# chunks = chunk_text(pdf_data["pages"])

# # Step 2: Generate embeddings
# _, embeddings = generate_embeddings(chunks)

# print("Embedding shape:", embeddings.shape)
# print("Sample embedding vector:", embeddings[0][:5])


import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks, batch_size=32):
    """
    Efficiently generate embeddings using BAAI/bge-small-en-v1.5.
    
    Args:
        chunks: List of dicts with 'text' (from chunker)
        batch_size: Batching for speed

    Returns:
        Tuple of:
            - Raw chunk texts
            - Embeddings as NumPy array
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return texts, embeddings


def generate_openai_embeddings(chunks, model_name="text-embedding-3-small", batch_size=32):
    """
    Generate embeddings using OpenAI's text-embedding-3-small model via LangChain.
    Args:
        chunks: List of dicts with 'text' (from chunker)
        model_name: OpenAI embedding model to use
        batch_size: Batching for speed (not all LangChain versions support this)
    Returns:
        Tuple of:
            - Raw chunk texts
            - Embeddings as a list or numpy array
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = OpenAIEmbeddings(model=model_name)  # Uses key from environment
    vectors = embeddings.embed_documents(texts)
    return texts, np.array(vectors, dtype=np.float32)


# from pdf_extractor import extract_pdf_content
# from chunker import chunk_text
# # from embedder_miniLM import generate_embeddings

# pdf_data = extract_pdf_content("dataset1.pdf")
# chunks = chunk_text(pdf_data["pages"])
# texts, embeddings = generate_embeddings(chunks)

# print("Total chunks:", len(texts))
# print("Embedding shape:", embeddings.shape)
