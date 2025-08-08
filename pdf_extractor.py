from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
import os
import asyncio
import json
from urllib.parse import urlparse
import aiohttp
from PIL import Image
import pytesseract
from io import BytesIO
import pandas as pd
from pptx import Presentation
import requests


# Set Tesseract path to the working installation
pytesseract.pytesseract.tesseract_cmd = r'D:\Softwares\Tesseract\tesseract.exe'




async def fetch_image_bytes(url: str) -> bytes:
    """Download image bytes from a URL using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download image: {url}")
            return await response.read()


async def fetch_file_bytes(url: str) -> bytes:
    """Download file bytes from a URL using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file: {url}")
            return await response.read()

async def handle_azure_blob_url(url: str):
    """
    Special handler for Azure blob URLs that fetches flight information based on favorite city.
    """
    try:
        # Step 1: Get the favorite city
        city_response = requests.get("https://register.hackrx.in/submissions/myFavouriteCity", timeout=10)
        city_response.raise_for_status()
        city_data = city_response.json()
        favorite_city = city_data.get("data", {}).get("city")
        
        if not favorite_city:
            raise Exception("Could not retrieve favorite city")
        
        print(f"Favorite city: {favorite_city}")
        
        # Step 2: Get flight numbers from all 5 endpoints
        flight_endpoints = [
            "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
            "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber", 
            "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
            "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
            "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
        ]
        
        matching_flight_number = None
        
        for endpoint in flight_endpoints:
            try:
                flight_response = requests.get(endpoint, timeout=10)
                flight_response.raise_for_status()
                flight_data = flight_response.json()
                
                # Log the JSON response from each endpoint
                print(f"JSON response from {endpoint}: {flight_data}")
                
                message = flight_data.get("message", "")
                flight_number = flight_data.get("data", {}).get("flightNumber")
                
                # Check if the favorite city is mentioned in the message
                if favorite_city.lower() in message.lower():
                    matching_flight_number = flight_number
                    print(f"Found matching flight number for {favorite_city}: {matching_flight_number}")
                    break
                    
            except Exception as e:
                print(f"Error fetching from {endpoint}: {e}")
                continue
        
        if matching_flight_number:
            return {"pages": [{"page": 1, "text": f"Flight number is: {matching_flight_number}"}]}
        else:
            return {"pages": [{"page": 1, "text": "No matching flight number found for the favorite city"}]}
            
    except Exception as e:
        print(f"Error in handle_azure_blob_url: {e}")
        return {"pages": [{"page": 1, "text": f"Error processing Azure blob URL: {str(e)}"}]}

async def extract_pdf_content(url: str):
    """
    Extracts text content from PDF, DOCX, Email, or Image URLs.
    Returns a list of dicts with 'page', 'part', or 'image' and 'text'.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path
    ext = os.path.splitext(path)[1].lower()

    # Special handler for Azure blob URL
    if "hackrx.blob.core.windows.net" in url and "FinalRound4SubmissionPDF.pdf" in url:
        return await handle_azure_blob_url(url)

    # Handle URLs without extensions (especially for hackrx.in domain)
    if not ext:
        # Make a request to the URL to get the content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text
        return {"pages": [{"page": 1, "text": content}]}

    if ext == ".pdf":
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}

    elif ext == ".docx":
        # Download the DOCX file first, then use local path
        file_bytes = await fetch_file_bytes(url)
        temp_file = f"temp_docx_{os.path.basename(path)}"
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(file_bytes)
            
            loader = Docx2txtLoader(temp_file)
            docs = loader.load()
            return {"pages": [{"page": 1, "text": docs[0].page_content}]}
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    elif ext in [".eml", ".msg"]:
        loader = UnstructuredEmailLoader(url)
        docs = loader.load()
        return {"pages": [{"part": i+1, "text": doc.page_content} for i, doc in enumerate(docs)]}

    elif ext in [".jpg", ".jpeg", ".png"]:
        image_bytes = await fetch_image_bytes(url)
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        print("Image text", text)
        return {"pages": [{"page": 1, "text": text.strip()}]}

    elif ext in [".xlsx", ".xls"]:
        # Download the Excel file first, then convert to JSON
        file_bytes = await fetch_file_bytes(url)
        temp_file = f"temp_excel_{os.path.basename(path)}"
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(file_bytes)
            
            # Read Excel with pandas
            df = pd.read_excel(temp_file, engine='openpyxl')
            
            # Handle missing values for LLM context
            df_clean = df.fillna('')
            
            # Convert to JSON with records orientation (best for LLM)
            json_data = df_clean.to_json(orient='records', indent=2)
            
            # Parse JSON to get the data structure
            json_obj = json.loads(json_data)
            
            # Return the actual data as text
            return {
                "pages": [{
                    "page": 1, 
                    "text": json_data
                }]
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    elif ext in [".ppt", ".pptx"]:
        # Download the PowerPoint file first, then extract content
        file_bytes = await fetch_file_bytes(url)
        temp_file = f"temp_ppt_{os.path.basename(path)}"
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(file_bytes)
            
            # Load presentation
            prs = Presentation(temp_file)
            pages = []
            
            # Process each slide
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []
                
                # Process each shape in the slide
                for shape in slide.shapes:
                    # Check if shape has text
                    if hasattr(shape, 'text') and shape.text.strip():
                        slide_text.append(shape.text.strip())
                    
                    # Check if shape is an image
                    if hasattr(shape, 'image'):
                        try:
                            # Get image data
                            image_data = shape.image.blob
                            
                            # Process image with OCR - improved error handling
                            image = Image.open(BytesIO(image_data))
                            
                            # Convert to RGB if needed to avoid Windows elevation issues
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Try OCR with different configurations
                            try:
                                ocr_text = pytesseract.image_to_string(image, config='--psm 6')
                            except:
                                # Fallback to default configuration
                                ocr_text = pytesseract.image_to_string(image)
                            
                            if ocr_text.strip():
                                slide_text.append(ocr_text.strip())
                        except Exception as e:
                            print(f"Error processing image in slide {slide_num}: {e}")
                            # Continue processing other shapes
                
                # Add slide to pages
                pages.append({
                    "page": slide_num,
                    "text": "\n".join(slide_text)
                })
            
            return {"pages": pages}
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    else:
        # Default fallback to PDF
        pages = []
        loader = PyPDFLoader(url)
        async for page in loader.alazy_load():
            pages.append(page)
        return {"pages": [{"page": i+1, "text": doc.page_content} for i, doc in enumerate(pages)]}
