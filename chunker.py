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
        # Handle different content types (PDF pages, images, email parts)
        if "page" in page:
            page_num = page["page"]
            content_type = "page"
        elif "image" in page:
            page_num = page["image"]  # Use image filename as identifier
            content_type = "image"
        elif "part" in page:
            page_num = page["part"]
            content_type = "part"
        else:
            # Fallback for unknown content types
            page_num = 1
            content_type = "unknown"
        
        text = page["text"]
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page_num,
                "content_type": content_type,
                "chunk_index": i
            })

    return all_chunks