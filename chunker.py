from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages, chunk_size=750, chunk_overlap=100):
    """
    Splits extracted PDF pages into overlapping text chunks.
    
    Args:
        pages: List of dicts with page text and metadata (from extract_pdf_content)
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between chunks (to preserve context)

    Returns:
        List of dicts: Each dict contains text chunk and metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for page in pages:
        page_num = page["page"]
        text = page["text"]
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page_num,
                "chunk_index": i
            })

    return all_chunks