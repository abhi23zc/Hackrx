# import pdfplumber
# import nltk
# import uuid
# from typing import List, Dict
# from nltk.tokenize import sent_tokenize
# import logging
# import re

# nltk.download("punkt")

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("pdf-preprocessor")

# # === CONFIGURABLE PARAMETERS === #
# MAX_TOKENS = 300
# OVERLAP = 50

# # === UTILITIES === #

# def tokenize(text: str) -> List[str]:
#     """Returns a list of tokens (basic whitespace tokenizer)."""
#     return text.split()

# def chunk_sentences(sentences: List[str], max_tokens=300, overlap=50) -> List[str]:
#     """Chunks sentence list into overlapping token-based blocks."""
#     chunks = []
#     current_chunk = []
#     current_len = 0

#     for sent in sentences:
#         token_len = len(tokenize(sent))
#         if current_len + token_len > max_tokens:
#             # Save current chunk
#             chunks.append(" ".join(current_chunk))

#             # Overlap logic
#             overlap_sents = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
#             current_chunk = overlap_sents
#             current_len = sum(len(tokenize(s)) for s in current_chunk)

#         current_chunk.append(sent)
#         current_len += token_len

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# def extract_section_title(text: str) -> str:
#     """
#     Extracts first heading-like phrase (e.g., 'Section 5:', '1.1') from text.
#     """
#     match = re.search(r'(Section\s?\d+[^\n:]*)|(^\d+(\.\d+)*\s[^\n]*)', text, re.IGNORECASE)
#     if match:
#         return match.group(0).strip()
#     return "Unknown Section"

# # === MAIN FUNCTION === #

# def preprocess_pdf(filepath: str) -> List[Dict]:
#     """
#     Extracts text from PDF and returns clean chunks with metadata, ready for vector DB or embedding.
#     """
#     all_chunks = []

#     try:
#         with pdfplumber.open(filepath) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 text = page.extract_text()

#                 if not text or not text.strip():
#                     logger.warning(f"Skipping empty or unextractable page {i + 1}")
#                     continue

#                 clean_text = text.strip().replace("\n", " ").replace("  ", " ")

#                 # Sentence split
#                 sentences = sent_tokenize(clean_text)
#                 if not sentences:
#                     continue

#                 # Optional: Extract section title from the first few sentences
#                 section_title = extract_section_title(sentences[0])

#                 # Chunk the page
#                 page_chunks = chunk_sentences(sentences, max_tokens=MAX_TOKENS, overlap=OVERLAP)

#                 for chunk in page_chunks:
#                     all_chunks.append({
#                         "id": str(uuid.uuid4()),
#                         "text": chunk,
#                         "meta": {
#                             "source_file": filepath.split("/")[-1],
#                             "page": i + 1,
#                             "section": section_title
#                         }
#                     })

#     except Exception as e:
#         logger.error(f"Failed to process {filepath}: {e}")

#     logger.info(f"Extracted {len(all_chunks)} chunks from {filepath}")
#     return all_chunks



# # if __name__ == "__main__":
# #     chunks = preprocess_pdf("./dataset1.pdf")
    
# #     # Preview chunks
# #     for chunk in chunks[:3]:
# #         print("="*60)
# #         print("Chunk ID:", chunk["id"])
# #         print("Section:", chunk["meta"]["section"])
# #         print("Page:", chunk["meta"]["page"])
# #         print("Text:\n", chunk["text"])

import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
import logging
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-preprocessor")

MAX_CHARS = 700
OVERLAP = 100

def preprocess_pdf(pdf_path: str):
    logger.info(f"Processing {pdf_path}...")
    doc = fitz.open(pdf_path)
    chunks = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        raw_text = page.get_text().strip()

        if not raw_text:
            continue

        sentences = sent_tokenize(raw_text)
        merged_text = ""
        buffer = []

        for sent in sentences:
            if len(merged_text) + len(sent) < MAX_CHARS:
                merged_text += " " + sent
                buffer.append(sent)
            else:
                # Save chunk
                chunk_text = merged_text.strip()
                chunks.append({
                    "id": str(uuid4()),
                    "text": chunk_text,
                    "meta": {
                        "page": page_number + 1,
                        "section": extract_section_heading(chunk_text),
                    }
                })
                # Slide window with overlap
                merged_text = " ".join(buffer[-(OVERLAP // 20):])  # Approx 20 chars per sentence
                buffer = buffer[-(OVERLAP // 20):] + [sent]

        # Add final leftover buffer
        if buffer:
            chunk_text = " ".join(buffer).strip()
            chunks.append({
                "id": str(uuid4()),
                "text": chunk_text,
                "meta": {
                    "page": page_number + 1,
                    "section": extract_section_heading(chunk_text),
                }
            })

    logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks

# === Optional Heuristic: Extract Section Heading
def extract_section_heading(text: str):
    lines = text.split("\n")
    for line in lines:
        if line.strip().istitle() and len(line.strip()) < 80:
            return line.strip()
    return "Unknown Section"
