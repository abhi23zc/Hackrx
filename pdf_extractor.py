from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
import os


def extract_pdf_content(file_path: str):
    """
    Extracts text content from PDF, DOCX, or Email files using LangChain loaders.
    Returns a list of dicts with 'page' (or 'part') and 'text'.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(docs)]}
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return {"pages": [{"page": 1, "text": docs[0].page_content}]}
    elif ext in [".eml", ".msg"]:
        loader = UnstructuredEmailLoader(file_path)
        docs = loader.load()
        return {"pages": [{"part": i+1, "text": doc.page_content} for i, doc in enumerate(docs)]}
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Example usage:
# if __name__ == "__main__":
#     file_path = "example.pdf"  # or .docx, .eml, .msg
#     result = extract_document_content(file_path)
#     print(result)
