from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
import os
import asyncio
from urllib.parse import urlparse
import aiohttp
from PIL import Image
import pytesseract
from io import BytesIO


# If tesseract.exe is in your current directory
# pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'




async def fetch_image_bytes(url: str) -> bytes:
    """Download image bytes from a URL using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download image: {url}")
            return await response.read()


async def extract_pdf_content(url: str):
    """
    Extracts text content from PDF, DOCX, Email, or Image URLs.
    Returns a list of dicts with 'page', 'part', or 'image' and 'text'.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}

    elif ext == ".docx":
        loader = Docx2txtLoader(url)
        docs = loader.load()
        return {"pages": [{"page": 1, "text": docs[0].page_content}]}

    elif ext in [".eml", ".msg"]:
        loader = UnstructuredEmailLoader(url)
        docs = loader.load()
        return {"pages": [{"part": i+1, "text": doc.page_content} for i, doc in enumerate(docs)]}

    elif ext in [".jpg", ".jpeg", ".png"]:
        image_bytes = await fetch_image_bytes(url)
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        print("Image text", text)
        return {"pages": [{"image": os.path.basename(path), "text": text.strip()}]}

    else:
        # Default fallback to PDF
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}
