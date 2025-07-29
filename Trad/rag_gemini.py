import google.generativeai as genai
from query_faiss import search
import logging
import json
import os
import re

# === Config ===
GEMINI_API_KEY = "AIzaSyAb2K0HUEY2b7lqcwE6qUrcxByxUN3D6ds"
TOP_K = 5

# === Init Gemini ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# === Logger ===
logger = logging.getLogger("RAG-Gemini")
logging.basicConfig(level=logging.INFO)

# === Extract JSON from code block (handles ```json ... ```)
def extract_json_from_code_block(text: str):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# === Build Prompt ===
def build_prompt(query: str, chunks: list[dict]) -> tuple[str, list[int]]:
    context_blocks = []
    source_pages = []

    for chunk in chunks:
        page = chunk['meta']['page']
        section = chunk['meta']['section']
        text = chunk['meta'].get("text", "[chunk text missing]")
        context_blocks.append(f"[Page {page}, Section: {section}]\n{text}")
        source_pages.append(page)

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are an intelligent legal-policy assistant for insurance documents.

Context:
{context_text}

User Question:
{query}

Instructions:
- Only answer using the context above.
- Do not guess or add external knowledge.
- If the answer is not found, respond with "Not mentioned".
- Keep it professional, structured, and use formal tone.

Output your response strictly as JSON:
{{
  "answer": "...",
  "source_pages": [list of page numbers],
  "rationale": "..."
}}
(Do not wrap this in backticks or markdown. Return only raw JSON.)
"""
    return prompt, list(set(source_pages))

# === Main RAG Handler ===
def answer_with_gemini(query: str, top_k=TOP_K):
    try:
        logger.info("🔍 Running semantic search...")
        top_chunks = search(query, top_k=top_k)

        prompt, source_pages = build_prompt(query, top_chunks)
        logger.info("✉️ Sending prompt to Gemini...")

        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        parsed = extract_json_from_code_block(raw_output)
        if parsed:
            logger.info("✅ Gemini returned valid JSON.")
            return parsed
        else:
            logger.warning("⚠️ Gemini response was not valid JSON. Returning raw output.")
            return {
                "answer": "Could not parse Gemini response as JSON.",
                "source_pages": source_pages,
                "rationale": raw_output
            }

    except Exception as e:
        logger.error(f"❌ RAG failed: {e}")
        return {
            "answer": "RAG system failed.",
            "source_pages": [],
            "rationale": str(e)
        }

# === CLI Test ===
if __name__ == "__main__":
    user_query = "Does the policy pay for rehabilitation costs after surgery?"
    result = answer_with_gemini(user_query)
    print(json.dumps(result, indent=2))

