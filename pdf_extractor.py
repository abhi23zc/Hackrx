from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
import asyncio
import os


async def extract_pdf_content(url: str):
    """
    Extracts text content from PDF, DOCX, or Email URLs using LangChain loaders.
    Returns a list of dicts with 'page' (or 'part') and 'text'.
    """
    # Check file extension from URL - handle query parameters properly
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    path = parsed_url.path
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".pdf":
        # Use PyPDFLoader directly with URL
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}
    elif ext == ".docx":
        # For DOCX from URL, we'd need to download first or use different approach
        # For now, assuming it's a local file path or we'll handle URL download
        loader = Docx2txtLoader(url)
        docs = loader.load()
        return {"pages": [{"page": 1, "text": docs[0].page_content}]}
    elif ext in [".eml", ".msg"]:
        # For email files from URL
        loader = UnstructuredEmailLoader(url)
        docs = loader.load()
        return {"pages": [{"part": i+1, "text": doc.page_content} for i, doc in enumerate(docs)]}
    else:
        # Default to PDF if no extension or unknown extension
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}

# Example usage:
# if __name__ == "__main__":
#     file_path = "example.pdf"  # or .docx, .eml, .msg
#     result = extract_document_content(file_path)
#     print(result)
