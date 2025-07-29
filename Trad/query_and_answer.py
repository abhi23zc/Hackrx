import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === CONFIG ===
MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX = "faiss_index.index"
FAISS_META = "faiss_index_meta.pkl"
TOP_K = 5

# === Load model, index, metadata
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX)

with open(FAISS_META, "rb") as f:
    metadata = pickle.load(f)

# === Step 1: Query → Embed → Search
def retrieve_relevant_chunks(query: str, top_k=TOP_K):
    query_embedding = model.encode("query: " + query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype("float32")
    scores, indices = index.search(query_embedding, top_k)

    chunks = []
    for i, idx in enumerate(indices[0]):
        chunks.append({
            "rank": i + 1,
            "score": float(scores[0][i]),
            "text": metadata[idx].get("text", "N/A"),
            "meta": metadata[idx]
        })

    return chunks

# === Step 2: Build Prompt and Ask LLM
def build_prompt(query: str, retrieved_chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Page {c['meta']['page']}] {c.get('text', '')}" for c in retrieved_chunks
    )

    return f"""You are an insurance policy analyst. Based on the context below, answer the user's question truthfully and concisely. If not answerable, say "Information not found."

User Query:
{query}

Context:
{context}

Answer:"""

def ask_llm(prompt: str, model="gpt-4"):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# === Combined Retrieval + QA
def rag_pipeline(query: str):
    retrieved = retrieve_relevant_chunks(query)
    prompt = build_prompt(query, retrieved)
    answer = ask_llm(prompt)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "page": chunk["meta"]["page"],
                "section": chunk["meta"]["section"],
                "score": chunk["score"]
            } for chunk in retrieved
        ]
    }

if __name__ == "__main__":
    query = "Does this policy cover knee surgery, and what are the conditions?"
    result = rag_pipeline(query)

    print("Answer:\n", result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- Page {src['page']} | Section: {src['section']} | Score: {src['score']:.4f}")
