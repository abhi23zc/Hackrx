import os
import asyncio
import aiohttp
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import logging
import time
import pickle
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
# Import our existing pipeline components
from pdf_extractor import extract_pdf_content
from chunker import chunk_text
from embedder import model, generate_embeddings, generate_openai_embeddings
from faiss_store import create_faiss_index, save_metadata
from retriever_reranker import retrieve_top_k, rerank_chunks
from prompt_builder import build_prompt_without_sources
import hashlib
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Add a function to hash the file link

def hash_filelink(filelink: str) -> str:
    return hashlib.sha256(filelink.encode('utf-8')).hexdigest()

# Remove EMBEDDINGS_DIR and all .pkl logic



# System prompts

GENERAL_SYSTEM_PROMPT = (
    "You are a HUMAN subject matter expert based strictly on the context of the provided document.\n"
    "These documents may include anything.\n\n"
    "CRITICAL RULES:\n"
    "- Each context chunk will have a similarity score in the format [Score: X.XXXX].\n"
    "- Higher simsilarity scores indicate more relevant information.\n"
    "- If the query is very clearly out of the domain of the provided context, that is all chunks have a SCORE less than 0.2, instantly return \"Question out of scope of the document\"\n"
    "- If after using the chunks and all analysis you can't find relevant information even with SCORE of all chunks more than 0.2,use general knowledge readily available on the net to answer the query imitating paraphrasing of the document.\n"
    " -After forming your answer if the query was a straight confirmational question, reframe the answer giving confirmation by the at max 2-3 facts in a precis fashion.\n"
    "- Use clear, professional language.\n"
    "- Focus on the core information requested.\n"
    "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources. Provide a focused answer with only essential details."
) 




class PDFRAGPipeline:
    def __init__(self):
        self.setup_groq()
        self.setup_openai()
        self.vector_store_path = "vector_store"
        self.embeddings_dir = "embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)
        # In-memory set of processed file hashes
        self.processed_hashes = set()
        # Populate set from existing .pkl files
        for fname in os.listdir(self.embeddings_dir):
            if fname.endswith('.pkl'):
                self.processed_hashes.add(fname[:-4])

    def setup_groq(self):
        """Configure OpenRouter API with Groq Llama3-70B-instruct"""
        try:
            # Get OpenRouter API key from environment
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OPENROUTER_API_KEY not set in environment. Groq LLM will not work.")
                raise ValueError("OPENROUTER_API_KEY environment variable is required")
            
            self.openrouter_llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct",
                openai_api_key=openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=1024
            )
            logger.info("✅ OpenRouter API configured successfully with Groq Llama3-70B-instruct")
        except Exception as e:
            logger.error(f"❌ Failed to configure OpenRouter: {e}")
            raise

    def setup_openai(self):
        """Configure OpenAI API client (via LangChain)"""
        try:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY not set in environment. OpenAI LLM will not work.")
            # Optionally, you could instantiate a ChatOpenAI client here if you want to reuse it
            # self.openai_llm = ChatOpenAI(
            #     model="gpt-3.5-turbo",
            #     temperature=0.1,
            #     max_tokens=1024,
            #     openai_api_key=self.openai_api_key
            # )
        except Exception as e:
            logger.error(f"❌ Failed to configure OpenAI: {e}")
            raise



    async def process_pdf(self, file_link: str) -> dict:
        """Process PDF content through the pipeline (URL only)."""
        try:
            file_hash = hash_filelink(file_link)
            pkl_path = os.path.join(self.embeddings_dir, f"{file_hash}.pkl")
            index_path = os.path.join(self.embeddings_dir, f"{file_hash}_index.faiss")
            meta_path = os.path.join(self.embeddings_dir, f"{file_hash}_metadata.json")
            # If already processed, load cached embeddings and skip download/processing
            if file_hash in self.processed_hashes and os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    result = pickle.load(f)
                # Recreate FAISS index and metadata if missing
                if not (os.path.exists(index_path) and os.path.exists(meta_path)):
                    os.makedirs(self.vector_store_path, exist_ok=True)
                    create_faiss_index(np.array(result["embeddings"]), index_path=index_path)
                    save_metadata(result["chunks"], meta_path=meta_path)
                return result
            
            # Not cached: process using URL directly
            logger.info("🔍 Extracting PDF content directly from URL...")
            logger.info(f"📄 URL being processed: {file_link}")
            try:
                pdf_data = await extract_pdf_content(file_link)
                logger.info("✅ PDF extraction completed successfully")
            except Exception as e:
                logger.error(f"❌ PDF extraction failed: {str(e)}")
                logger.error(f"❌ Error type: {type(e)}")
                raise
            
            logger.info(f"✅ Extracted {len(pdf_data['pages'])} pages")
            logger.info("✂️ Chunking text...")
            chunks = chunk_text(pdf_data["pages"])
            logger.info(f"✅ Created {len(chunks)} chunks")
            logger.info("🧠 Generating embeddings with HuggingFace model...")
            texts, embeddings = generate_embeddings(chunks)
            logger.info(f"✅ Generated embeddings: {getattr(embeddings, 'shape', type(embeddings))}")
            logger.info("💾 Storing in vector database...")
            os.makedirs(self.vector_store_path, exist_ok=True)
            create_faiss_index(
                embeddings,
                index_path=index_path
            )
            save_metadata(
                chunks,
                meta_path=meta_path
            )
            logger.info("✅ Vector store created successfully")
            result = {
                "success": True,
                "chunks": chunks,
                "pages": len(pdf_data['pages']),
                "embeddings": embeddings,
                "texts": texts
            }
            # Save embeddings to .pkl and update set
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(result, f)
                self.processed_hashes.add(file_hash)
            except Exception as e:
                logger.error(f"Error saving embeddings to {pkl_path}: {e}")
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )

    async def answer_questions(self, questions: List[str], llm_provider: str = "groq", file_hash: str = None, request_start_time: float = None) -> List[str]:
        """Answer questions using the selected LLM provider with robust timeout mechanism."""
        try:
            # Initialize answers array with empty strings to match questions length
            answers = [""] * len(questions)
            # Use request start time if provided, otherwise use current time
            start_time = request_start_time if request_start_time is not None else time.time()
            timeout_threshold = 29.0  # 29 seconds threshold
            
            # Calculate remaining time
            elapsed_time = time.time() - start_time
            remaining_time = max(0, timeout_threshold - elapsed_time)
            
            if remaining_time <= 0:
                logger.info(f"⏰ No time remaining ({elapsed_time:.2f}s elapsed). Returning empty answers.")
                return answers
            
            # Process questions concurrently with robust timeout
            tasks = []
            for i, question in enumerate(questions):
                task = asyncio.create_task(self._process_single_question_with_index(i, question, llm_provider, file_hash, answers, start_time, timeout_threshold))
                tasks.append(task)
            
            # Use asyncio.wait_for with timeout to actually interrupt tasks
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=remaining_time)
                logger.info("✅ All questions processed within time limit")
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout reached ({timeout_threshold:.2f}s). Cancelling remaining tasks.")
                # Cancel any remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait a bit for cancellations to take effect
                await asyncio.sleep(0.1)
            
            return answers
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error answering questions: {str(e)}"
            )

    async def _process_single_question_with_index(self, index: int, question: str, llm_provider: str, file_hash: str, answers: List[str], start_time: float, timeout_threshold: float):
        """Process a single question and update the answers list at the specified index."""
        try:
            # Check if we're approaching the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_threshold:
                logger.info(f"⏰ Timeout threshold reached ({elapsed_time:.2f}s). Skipping question {index + 1}")
                answers[index] = "Skipping question adhering to 30s response time limits"
                return
            
            logger.info(f"🔍 Processing question {index + 1}: {question} (LLM: {llm_provider})")
            
            # Step 1: Retrieve relevant chunks using hash-based paths
            index_path = os.path.join(self.embeddings_dir, f"{file_hash}_index.faiss")
            meta_path = os.path.join(self.embeddings_dir, f"{file_hash}_metadata.json")
            
            retrieved = retrieve_top_k(question, k=5, index_path=index_path, meta_path=meta_path)
            logger.info(f"✅ Retrieved {len(retrieved)} relevant chunks for question {index + 1}")
            
            # Check for cancellation after retrieval
            if asyncio.current_task().cancelled():
                logger.info(f"🛑 Question {index + 1} cancelled after retrieval")
                return
            
            if not retrieved:
                logger.error(f"No relevant chunks retrieved for question {index + 1}.")
                answers[index] = "Information not available in the provided document."
                return
            
            # Step 2: Rerank chunks
            reranked = rerank_chunks(question, retrieved, top_n=3)
            logger.info(f"✅ Reranked to top {len(reranked)} chunks for question {index + 1}")
            
            # Check for cancellation after reranking
            if asyncio.current_task().cancelled():
                logger.info(f"🛑 Question {index + 1} cancelled after reranking")
                return
            
            if not reranked:
                logger.error(f"No chunks after reranking for question {index + 1}.")
                answers[index] = "Information not available in the provided document."
                return
            
            # Step 3: Build optimized prompt with similarity scores
            prompt = build_prompt_without_sources(question, reranked)
            system_prompt = GENERAL_SYSTEM_PROMPT
            
            # Step 4: Get answer from the selected LLM
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Check for cancellation before LLM call
                    if asyncio.current_task().cancelled():
                        logger.info(f"🛑 Question {index + 1} cancelled before LLM call")
                        return
                    
                    # Check timeout before each attempt
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout_threshold:
                        logger.info(f"⏰ Timeout threshold reached ({elapsed_time:.2f}s). Stopping question {index + 1}")
                        answers[index] = "Skipping question adhering to 30s response time limits"
                        return
                    
                    if llm_provider == "openai":
                        if not self.openai_api_key:
                            logger.error("OPENAI_API_KEY not set. Cannot use OpenAI LLM.")
                            answers[index] = "OpenAI API key not configured."
                            return
                        openai_llm = ChatOpenAI(
                            model="gpt-4o",
                            temperature=0.1,
                            max_tokens=1024,
                            openai_api_key=self.openai_api_key
                        )
                        response = await openai_llm.ainvoke([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ])
                        answer = response.content.strip()
                    else:
                        response = await self.openrouter_llm.ainvoke([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ])
                        answer = response.content.strip()
                    
                    # Update the answers list at the correct index
                    answers[index] = answer
                    logger.info(f"✅ Generated answer for question {index + 1}")
                    return
                    
                except Exception as e:
                    # Check for 429 error
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = min(2 ** attempt, 5)  # Cap wait time at 5 seconds
                        logger.warning(f"Rate limited by LLM API for question {index + 1}. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Error processing question {index + 1}: {str(e)}")
                        answers[index] = "Unable to process this question at the moment."
                        return
            
            # If we get here, all retries failed
            answers[index] = "Unable to process this question at the moment."
            
        except Exception as e:
            logger.error(f"Error processing question {index + 1}: {repr(e)}", exc_info=True)
            answers[index] = "Unable to process this question at the moment."

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
    # Start timer immediately when request comes in
    request_start_time = time.time()
    logger.info(f"⏰ Request timer started at: {request_start_time}")

    try:
        logger.info(f"📄 Processing request for PDF: {request.documents}")
        logger.info(f"📝 Questions to answer: {request.questions}")
        
        file_hash = hash_filelink(str(request.documents))
        logger.info(f"🔗 Generated file hash: {file_hash}")
        
        pkl_path = os.path.join(pipeline.embeddings_dir, f"{file_hash}.pkl")
        logger.info(f"📁 Checking cache at: {pkl_path}")
        
        # Optimization: Check if hash is already processed before downloading
        if file_hash in pipeline.processed_hashes and os.path.exists(pkl_path):
            logger.info(f"⚡ Cache hit for document hash: {file_hash}. Using cached embeddings, skipping download and processing.")
            with open(pkl_path, "rb") as f:
                process_result = pickle.load(f)
            logger.info("✅ Loaded cached embeddings successfully")
        else:
            logger.info("🔄 Cache miss - Processing PDF through pipeline...")
            process_result = await pipeline.process_pdf(file_link=str(request.documents))
            logger.info("✅ PDF processing completed")
        
        # Answer questions using Groq with timer-based cutoff
        logger.info("🤖 Answering questions with Groq (timer-based)...")
        answers = await pipeline.answer_questions(request.questions, llm_provider="groq", file_hash=file_hash, request_start_time=request_start_time)
        
        # Cleanup vector store
        import shutil
        if os.path.exists(pipeline.vector_store_path):
            shutil.rmtree(pipeline.vector_store_path)
        
        total_elapsed_time = time.time() - request_start_time
        logger.info(f"✅ Request completed in {total_elapsed_time:.2f} seconds")
        logger.info(f"📊 Final answers count: {len(answers)} (expected: {len(request.questions)})")
        
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