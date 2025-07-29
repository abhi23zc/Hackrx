import os
import io
import asyncio
import aiohttp
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import logging
import time

# Import our existing pipeline components
from pdf_extractor import extract_pdf_content
from chunker import chunk_text
from embedder import generate_embeddings
from faiss_store import create_faiss_index, save_metadata
from retriever_reranker import retrieve_top_k, rerank_chunks
from prompt_builder import build_prompt_without_sources

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="HackRx PDF RAG API", version="1.0.0")

# Security
security = HTTPBearer()

# API Key validation
VALID_API_KEY = "hackrx-2024-secure-key"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API key from Bearer token"""
    if credentials.credentials != VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return credentials.credentials

# Request/Response models
class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# Global pipeline state
class PDFRAGPipeline:
    def __init__(self):
        self.setup_gemini()
        self.vector_store_path = "temp_vector_store"
        
    def setup_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key="AIzaSyAb2K0HUEY2b7lqcwE6qUrcxByxUN3D6ds")
            self.model = genai.GenerativeModel("gemini-2.5-pro")
            logger.info("✅ Gemini API configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini: {e}")
            raise
    
    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download PDF: HTTP {response.status}"
                        )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error downloading PDF: {str(e)}"
            )
    
    async def process_pdf(self, pdf_content: bytes) -> dict:
        """Process PDF content through the pipeline"""
        try:
            # Save PDF temporarily
            temp_pdf_path = "temp_document.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)
            
            # Step 1: Extract PDF content
            logger.info("🔍 Extracting PDF content...")
            pdf_data = extract_pdf_content(temp_pdf_path)
            logger.info(f"✅ Extracted {len(pdf_data['pages'])} pages")
            
            # Step 2: Chunk the text
            logger.info("✂️ Chunking text...")
            chunks = chunk_text(pdf_data["pages"])
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("🧠 Generating embeddings...")
            texts, embeddings = generate_embeddings(chunks)
            logger.info(f"✅ Generated embeddings: {embeddings.shape}")
            
            # Step 4: Store in FAISS
            logger.info("💾 Storing in vector database...")
            os.makedirs(self.vector_store_path, exist_ok=True)
            create_faiss_index(
                embeddings, 
                index_path=os.path.join(self.vector_store_path, "index.faiss")
            )
            save_metadata(
                chunks, 
                meta_path=os.path.join(self.vector_store_path, "metadata.json")
            )
            logger.info("✅ Vector store created successfully")
            
            # Cleanup
            os.remove(temp_pdf_path)
            
            return {
                "success": True,
                "chunks": chunks,
                "pages": len(pdf_data['pages'])
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
    
    async def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer questions based on processed document"""
        try:
            answers = []
            
            for question in questions:
                logger.info(f"🔍 Processing question: {question}")
                
                # Step 1: Retrieve relevant chunks
                retrieved = retrieve_top_k(question, k=10)
                logger.info(f"✅ Retrieved {len(retrieved)} relevant chunks")
                
                # Step 2: Rerank chunks
                reranked = rerank_chunks(question, retrieved, top_n=5)
                logger.info(f"✅ Reranked to top {len(reranked)} chunks")
                
                # Step 3: Build prompt and generate answer
                prompt = build_prompt_without_sources(question, reranked)
                
                # Step 4: Get answer from Gemini
                response = self.model.generate_content(prompt)
                answer = response.text.strip()
                answers.append(answer)
                
                logger.info(f"✅ Generated answer for question")
            
            return answers
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error answering questions: {str(e)}"
            )

# Initialize pipeline
pipeline = PDFRAGPipeline()

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_questions(
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process PDF document and answer questions
    
    - **documents**: URL to PDF document
    - **questions**: List of questions to answer
    """
    start_time = time.time()
    
    try:
        # Download PDF
        logger.info(f"📄 Downloading PDF from: {request.documents}")
        pdf_content = await pipeline.download_pdf(str(request.documents))
        
        # Process PDF
        logger.info("🔄 Processing PDF through pipeline...")
        process_result = await pipeline.process_pdf(pdf_content)
        
        # Answer questions
        logger.info("🤖 Answering questions...")
        answers = await pipeline.answer_questions(request.questions)
        
        # Cleanup vector store
        import shutil
        if os.path.exists(pipeline.vector_store_path):
            shutil.rmtree(pipeline.vector_store_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed_time:.2f} seconds")
        
        return AnswerResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hackrx-rag-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=None,  # Add your SSL certificates for HTTPS
        ssl_certfile=None
    )