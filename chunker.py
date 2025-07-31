from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages, chunk_size=1000, chunk_overlap=500):
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

# --- New Semantic Chunker ---
def semantic_chunk_text(pages, model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=50):
    """
    Splits extracted PDF pages into semantic chunks using LangChain's SemanticChunker.
    Args:
        pages: List of dicts with page text and metadata (from extract_pdf_content)
        model_name: Name of the HuggingFace model to use for embeddings
        chunk_size: Target size for each chunk (in characters or tokens, as supported)
        chunk_overlap: Overlap between chunks (if supported)
    Returns:
        List of dicts: Each dict contains text chunk and metadata
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    semantic_splitter = SemanticChunker(embeddings)
    all_chunks = []
    for page in pages:
        page_num = page["page"]
        text = page["text"]
        chunks = semantic_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page_num,
                "chunk_index": i
            })
    return all_chunks


# from pdf_extractor import extract_pdf_content
# from chunker import chunk_text

# pdf_data = extract_pdf_content("dataset1.pdf")
# chunks = chunk_text(pdf_data["pages"])

# print(f"Total chunks: {len(chunks)}")
# print("Sample chunk:", chunks[0])