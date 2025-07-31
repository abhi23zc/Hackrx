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
from groq import AsyncGroq
import json
import pickle  # NEW: for saving/loading embeddings
import re      # NEW: for sanitizing filenames

# Import our existing pipeline components
from pdf_extractor import extract_pdf_content
from chunker import chunk_text
from embedder import generate_embeddings
from faiss_store import create_faiss_index, save_metadata
from retriever_reranker import retrieve_top_k, rerank_chunks
from prompt_builder import build_prompt_without_sources

# Remove Google imports
# import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="HackRx PDF RAG API", version="1.0.0")

# Security
security = HTTPBearer()

# API Key validation
VALID_API_KEY = "2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff"

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

EMBEDDINGS_DIR = "embeddings"  # NEW: directory for .pkl files

def sanitize_filename(file_link: str) -> str:
    """Sanitize file link to create a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', file_link)

# System prompts
INSURANCE_SYSTEM_PROMPT = (
    "You are a specialized AI assistant for health insurance policy analysis. Provide precise, factual answers based on the policy document.\n\n"
    "CRITICAL RULES:\n"
    "- Answer exactly what is asked with the most important details only\n"
    "- Include specific numbers, time periods, and key conditions\n"
    "- Keep answers to 1-2 sentences maximum\n"
    "- Use clear, professional language\n"
    "- Focus on the core information requested\n"
    "- If information is not in the context, respond with: \"Information not available in the provided document.\"\n\n"
    "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources. Provide a focused answer with only the essential policy details that directly answer the question."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a specialized AI assistant designed to provide precise, factual answers based strictly on the context of the provided document. "
    "These documents may include insurance policies, legal contracts, HR manuals, compliance guidelines, technical manuals, brochures, academic materials, or other large, unstructured texts.\n\n"
    "CRITICAL RULES:\n"
    "- Answer exactly what is asked with the most important details only\n"
    "- Include specific numbers, time periods, names, or key conditions when relevant\n"
    "- Keep answers to 1-2 sentences maximum\n"
    "- Use clear, professional language\n"
    "- Focus on the core information requested\n"
    "- If information is not in the context, respond with: \"Information not available in the provided document.\"\n\n"
    "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources. Provide a focused answer with only the essential details from the document that directly answer the question."
)

class PDFRAGPipeline:
    def __init__(self):
        self.setup_groq()
        self.vector_store_path = "vector_store"
        # NEW: Ensure embeddings directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        # NEW: In-memory set of processed file links
        self.processed_links = set()
        # Populate set from existing .pkl files
        for fname in os.listdir(EMBEDDINGS_DIR):
            if fname.endswith('.pkl'):
                # Reverse sanitize to get file link if needed, or just store sanitized names
                self.processed_links.add(fname[:-4])

    def setup_groq(self):
        """Configure Groq API with efficient settings"""
        try:
            # Use environment variable for API key
            groq_api_key = os.getenv("GROQ_API_KEY", "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")
            self.groq_client = AsyncGroq(api_key=groq_api_key)
            self.model_name = "llama-3.3-70b-versatile"  # Most efficient model
            logger.info("✅ Groq API configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Groq: {e}")
            raise

    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL (unchanged)"""
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

    async def process_pdf(self, pdf_content: bytes, file_link: str = None) -> dict:
        """Process PDF content through the pipeline (optimized)."""
        try:
            sanitized = sanitize_filename(file_link) if file_link else None
            pkl_path = os.path.join(EMBEDDINGS_DIR, f"{sanitized}.pkl") if sanitized else None
            # Removed: If file_link is provided and .pkl exists, load and return cached result
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
            # Step 3: Generate embeddings (batch processing)
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
            result = {
                "success": True,
                "chunks": chunks,
                "pages": len(pdf_data['pages']),
                "embeddings": embeddings,
                "texts": texts
            }
            # Save embeddings to .pkl and update set
            if sanitized and pkl_path:
                with open(pkl_path, "wb") as f:
                    pickle.dump(result, f)
                self.processed_links.add(sanitized)
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )

    async def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer questions using Groq with optimized batch processing"""
        try:
            answers = []

            # Process questions in parallel for efficiency
            tasks = [self._process_single_question(question) for question in questions]
            answers = await asyncio.gather(*tasks)

            return answers

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error answering questions: {str(e)}"
            )

    async def _process_single_question(self, question: str) -> str:
        """Process a single question with Groq"""
        try:
            logger.info(f"🔍 Processing question: {question}")

            # Step 1: Retrieve relevant chunks
            retrieved = retrieve_top_k(question, k=5)
            logger.info(f"✅ Retrieved {len(retrieved)} relevant chunks")

            # Step 2: Rerank chunks
            reranked = rerank_chunks(question, retrieved, top_n=3)
            logger.info(f"✅ Reranked to top {len(reranked)} chunks")

            # Step 3: Build optimized prompt
            prompt = build_prompt_without_sources(question, reranked)

            # Choose which system prompt to use
            system_prompt = GENERAL_SYSTEM_PROMPT  # Change to INSURANCE_SYSTEM_PROMPT if needed

            # Step 4: Get answer from Groq with optimized settings
            response = await self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.9,
                stream=False
            )

            answer = response.choices[0].message.content.strip()
            logger.info("✅ Generated answer for question")

            return answer

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I encountered an error answering this question: {str(e)}"

# Initialize pipeline
pipeline = PDFRAGPipeline()

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def process_questions(
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process PDF document and answer questions using Groq
    
    - **documents**: URL to PDF document
    - **questions**: List of questions to answer
    """
    start_time = time.time()

    try:
        # Sanitize file link for cache
        sanitized = sanitize_filename(str(request.documents))
        pkl_path = os.path.join(EMBEDDINGS_DIR, f"{sanitized}.pkl")
        # If already processed, load cached embeddings and skip download/processing
        if sanitized in pipeline.processed_links and os.path.exists(pkl_path):
            logger.info(f"⚡ Using cached embeddings for {request.documents}")
            with open(pkl_path, "rb") as f:
                process_result = pickle.load(f)
            # Recreate FAISS index and metadata if missing
            index_path = os.path.join(pipeline.vector_store_path, "index.faiss")
            meta_path = os.path.join(pipeline.vector_store_path, "metadata.json")
            import numpy as np
            from faiss_store import create_faiss_index, save_metadata
            if not (os.path.exists(index_path) and os.path.exists(meta_path)):
                logger.info("♻️ Recreating FAISS index and metadata from cache...")
                os.makedirs(pipeline.vector_store_path, exist_ok=True)
                create_faiss_index(np.array(process_result["embeddings"]), index_path=index_path)
                save_metadata(process_result["chunks"], meta_path=meta_path)
        else:
            # Download PDF
            logger.info(f"📄 Downloading PDF from: {request.documents}")
            pdf_content = await pipeline.download_pdf(str(request.documents))
            # Process PDF (with cache logic)
            logger.info("🔄 Processing PDF through pipeline...")
            process_result = await pipeline.process_pdf(pdf_content, file_link=str(request.documents))
        # Answer questions using Groq
        logger.info("🤖 Answering questions with Groq...")
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
    return {"status": "healthy", "service": "hackrx-rag-api", "ai_provider": "groq"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=None,
        ssl_certfile=None
    )