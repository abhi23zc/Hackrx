import fitz  # PyMuPDF

def extract_pdf_content(file_path: str):
    """
    Efficiently extract text from a PDF using PyMuPDF (fitz),
    returning per-page structured chunks.
    """
    doc = fitz.open(file_path)
    page_chunks = []
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
            if text.strip():
                page_chunks.append({
                    "page": page_num,
                    "text": text.strip()
                })
        except Exception as e:
            print(f"Warning: Failed to extract page {page_num}: {e}")
            continue
    doc.close()
    return {
        "pages": page_chunks
    }

# if __name__ == "__main__":
#     file_path = "dataset1.pdf"
#     result = extract_pdf_content(file_path)
#     print("First page text preview:\n", result["pages"][0]["text"][:300])
