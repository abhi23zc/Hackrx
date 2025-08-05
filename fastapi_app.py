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
from pymongo import MongoClient

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

class QASaveRequest(BaseModel):
    url: HttpUrl
    questions: List[str]
    answers: List[str]

class QASaveResponse(BaseModel):
    success: bool
    message: str
    document_id: str

# Add a function to hash the file link

def hash_filelink(filelink: str) -> str:
    return hashlib.sha256(filelink.encode('utf-8')).hexdigest()

def hash_question(question: str) -> str:
    """Generate a hash for a question string."""
    # Normalize the question by stripping whitespace and converting to lowercase
    normalized_question = question.strip().lower()
    return hashlib.sha256(normalized_question.encode('utf-8')).hexdigest()

# Remove EMBEDDINGS_DIR and all .pkl logic



# System prompts

GENERAL_SYSTEM_PROMPT = (
    "You are a HUMAN subject matter expert based strictly on the context of the provided document.\n"
    "These documents may include anything.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- Each context chunk will have a similarity score in the format [Score: X.XXXX].\n"
    "- Higher similarity scores indicate more relevant information.\n"
    "- If the query is very clearly out of the domain of the provided context, that is all chunks have a SCORE less than 0.2, instantly return \"Question out of scope of the document\"\n"
    "- If after using the context and all analysis you can't find relevant information, use general knowledge readily available on the internet to answer the query, answering it as if you are answering strictly from the document. Do not say couldn't find in context.\n"
    "- After forming your answer, rephrase it so it means the same but in 2-3 grammatically correct sentences of 8–15 words each.\n"
    "- Do NOT include any breakpoint characters like '\\n' in your answer.\n"
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
            
            # Log the final prompt being sent to LLM for main processing
            logger.info(f"🤖 FINAL PROMPT FOR MAIN PROCESSING (Question {index + 1}):")
            logger.info(f"System: {system_prompt}")
            logger.info(f"User: {prompt}")
            
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
                            temperature=0.5,
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

async def rephrase_cached_answers(questions: List[str], cached_answers: List[str], request_start_time: float = None) -> List[str]:
    """
    Rephrase cached answers using LLM while maintaining the same meaning.
    Uses concurrent processing like the original answer_questions method.
    
    Args:
        questions (List[str]): List of questions
        cached_answers (List[str]): List of cached answers
        request_start_time (float): Start time for timeout calculation
        
    Returns:
        List[str]: List of rephrased answers
    """
    # Initialize answers array with empty strings to match questions length
    rephrased_answers = [""] * len(questions)
    
    # Use request start time if provided, otherwise use current time
    start_time = request_start_time if request_start_time is not None else time.time()
    timeout_threshold = 29.0  # 29 seconds threshold (same as original)
    
    # Calculate remaining time
    elapsed_time = time.time() - start_time
    remaining_time = max(0, timeout_threshold - elapsed_time)
    
    if remaining_time <= 0:
        logger.info(f"⏰ No time remaining for rephrasing ({elapsed_time:.2f}s elapsed). Returning original answers.")
        return cached_answers
    
    # Process questions concurrently like the original method
    tasks = []
    for i, (question, cached_answer) in enumerate(zip(questions, cached_answers)):
        if cached_answer == "":  # Skip empty answers (not found in cache)
            continue
        task = asyncio.create_task(_rephrase_single_answer_with_index(i, question, cached_answer, rephrased_answers, start_time, timeout_threshold))
        tasks.append(task)
    
    # Use asyncio.wait_for with timeout to actually interrupt tasks (same as original)
    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=remaining_time)
            logger.info("✅ All cached answers rephrased within time limit")
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout reached for rephrasing ({timeout_threshold:.2f}s). Cancelling remaining tasks.")
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait a bit for cancellations to take effect
            await asyncio.sleep(0.1)
    
    return rephrased_answers

async def _rephrase_single_answer_with_index(index: int, question: str, cached_answer: str, rephrased_answers: List[str], start_time: float, timeout_threshold: float):
    """Rephrase a single cached answer and update the answers list at the specified index."""
    try:
        # Check if we're approaching the timeout (same as original)
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout_threshold:
            logger.info(f"⏰ Timeout threshold reached ({elapsed_time:.2f}s). Skipping rephrasing for question {index + 1}")
            rephrased_answers[index] = cached_answer  # Use original answer
            return
        
        logger.info(f"🔄 Rephrasing cached answer for question {index + 1}")
        
        # Create prompt for rephrasing
        rephrase_prompt = f"""Please rephrase the following answer while maintaining its exact meaning and accuracy. 
        Keep the same level of detail and information, but use different wording and sentence structure.
        
        IMPORTANT: Respond with ONLY the rephrased answer text. Do not include any labels, prefixes, or additional text like "Rephrased Answer:" or "Here's the rephrased version:". Just provide the rephrased statement directly.

        Question: {question}
        
        Original Answer: {cached_answer}"""
        
        # Log the final prompt being sent to LLM for rephrasing
        logger.info(f"🤖 FINAL PROMPT FOR REPHRASING (Question {index + 1}):")
        logger.info(f"System: You are a helpful assistant that rephrases text while maintaining exact meaning and accuracy. Do not add, remove, or change any factual information. IMPORTANT: Always respond with ONLY the rephrased text, no labels, prefixes, or additional formatting.")
        logger.info(f"User: {rephrase_prompt}")
        
        # Check for cancellation before LLM call (same as original)
        if asyncio.current_task().cancelled():
            logger.info(f"🛑 Rephrasing for question {index + 1} cancelled before LLM call")
            rephrased_answers[index] = cached_answer
            return
        
        # Use the same LLM as the main pipeline
        response = await pipeline.openrouter_llm.ainvoke([
            {"role": "system", "content": "You are a helpful assistant that rephrases text while maintaining exact meaning and accuracy. Do not add, remove, or change any factual information. IMPORTANT: Always respond with ONLY the rephrased text, no labels, prefixes, or additional formatting."},
            {"role": "user", "content": rephrase_prompt}
        ])
        
        
        rephrased_answer = response.content.strip()
        rephrased_answers[index] = rephrased_answer
        logger.info(f"✅ Rephrased answer for question {index + 1}")
        
    except Exception as e:
        logger.error(f"❌ Error rephrasing answer for question {index + 1}: {str(e)}")
        # Fallback to original answer if rephrasing fails
        rephrased_answers[index] = cached_answer

async def check_mongodb_for_answers(file_hash: str, questions: List[str]) -> List[str]:
    """
    Check MongoDB for existing answers for the given file hash and questions.
    Returns answers for questions that are found, empty strings for those not found.
    
    Args:
        file_hash (str): Hash of the file URL
        questions (List[str]): List of questions to check
        
    Returns:
        List[str]: List of answers (empty string if question not found), None if no document exists
    """
    try:
        # MongoDB connection configuration
        mongo_username = os.getenv("MONGO_USERNAME", "ansh")
        mongo_password = os.getenv("MONGO_PASSWORD", "jaiswal")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "HackRX")
        mongo_host = os.getenv("MONGO_HOST", "127.0.0.1")
        mongo_port = int(os.getenv("MONGO_PORT", "27017"))
        
        # Create MongoDB connection string
        connection_string = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db_name}?retryWrites=true"
        
        client = None
        try:
            # Connect to MongoDB
            client = MongoClient(connection_string)
            
            # Test the connection
            client.admin.command('ping')
            
            # Get database and collection
            db = client[mongo_db_name]
            collection = db["qa_pairs"]
            
            # Find the document with the given file hash
            document = collection.find_one({"_id": file_hash})
            
            if not document:
                logger.info(f"❌ No document found in MongoDB for hash: {file_hash}")
                return None
            
            # Check if all questions exist in the document
            stored_answers = document.get("answers", [])
            stored_question_hashes = document.get("question_hashes", [])
            
            if len(stored_question_hashes) != len(stored_answers):
                logger.warning(f"⚠️ Mismatch in question_hashes/answers count for hash: {file_hash}")
                return None
            
            # Create a mapping of question hashes to answers for fast lookup
            qa_mapping = dict(zip(stored_question_hashes, stored_answers))
            
            # Check each question individually and build answers list
            answers = []
            found_count = 0
            
            for i, question in enumerate(questions):
                question_hash = hash_question(question)
                
                # Use hash lookup only
                if question_hash in qa_mapping:
                    answers.append(qa_mapping[question_hash])
                    found_count += 1
                    logger.info(f"✅ Question {i+1} found in cache: {question[:50]}...")
                else:
                    answers.append("")  # Empty string for questions not found
                    logger.info(f"❌ Question {i+1} not found in cache: {question[:50]}...")
            
            if found_count > 0:
                logger.info(f"✅ Found {found_count}/{len(questions)} questions in MongoDB cache")
                return answers
            else:
                logger.info("❌ No questions found in cache")
                return None
                
        except Exception as e:
            logger.error(f"❌ MongoDB connection error while checking cache: {str(e)}")
            return None
        finally:
            # Always close the MongoDB connection
            if client:
                client.close()
                
    except Exception as e:
        logger.error(f"❌ Error checking MongoDB cache: {str(e)}")
        return None

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
        # Log the complete request JSON body
        logger.info("📥 REQUEST JSON BODY:")
        logger.info(f"documents: {request.documents}")
        logger.info(f"questions: {request.questions}")
        
        logger.info(f"📄 Processing request for PDF: {request.documents}")
        logger.info(f"📝 Questions to answer: {request.questions}")
        
        file_hash = hash_filelink(str(request.documents))
        logger.info(f"🔗 Generated file hash: {file_hash}")
        
        # First, check if answers already exist in MongoDB
        logger.info("🔍 Checking MongoDB for existing QA data...")
        cached_answers = await check_mongodb_for_answers(file_hash, request.questions)
        
        if cached_answers is None:
            logger.info("❌ No document found in MongoDB, proceeding with full PDF processing...")
            # No document exists, process all questions
            questions_to_process = [(i, q) for i, q in enumerate(request.questions)]
            final_answers = [""] * len(request.questions)
        else:
            # Document exists, check which questions need processing
            questions_to_process = []
            final_answers = cached_answers.copy()
            
            for i, (question, cached_answer) in enumerate(zip(request.questions, cached_answers)):
                if cached_answer == "":  # Question not found in cache
                    questions_to_process.append((i, question))
            
            if not questions_to_process:
                logger.info("✅ All questions found in MongoDB cache")
                # Don't return immediately - continue to apply minimum response time
            else:
                logger.info(f"🔄 Found {len(cached_answers) - len(questions_to_process)}/{len(request.questions)} questions in cache, processing {len(questions_to_process)} remaining questions...")
        
        # Rephrase cached answers using LLM
        if cached_answers is not None:
            logger.info("🔄 Rephrasing cached answers using LLM...")
            rephrased_answers = await rephrase_cached_answers(request.questions, cached_answers, request_start_time)
            final_answers = rephrased_answers
        
        # Process remaining questions that weren't found in cache
        if questions_to_process:
            # Extract just the questions (without indices) for processing
            questions_to_process_list = [q for _, q in questions_to_process]
            question_indices = [i for i, _ in questions_to_process]
            
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
            
            # Answer only the remaining questions using Groq with timer-based cutoff
            logger.info(f"🤖 Answering {len(questions_to_process_list)} remaining questions with Groq (timer-based)...")
            processed_answers = await pipeline.answer_questions(questions_to_process_list, llm_provider="groq", file_hash=file_hash, request_start_time=request_start_time)
            
            # Update final_answers with the processed results
            for idx, answer in zip(question_indices, processed_answers):
                final_answers[idx] = answer
            
            # Cleanup vector store
            import shutil
            if os.path.exists(pipeline.vector_store_path):
                shutil.rmtree(pipeline.vector_store_path)
        else:
            # All questions were found in cache, no processing needed
            logger.info("✅ All questions found in cache, no processing required")
        
        # Calculate minimum response time based on total questions
        total_questions = len(request.questions)
        if total_questions <= 3:
            min_response_time = 5.0
        elif total_questions <= 7:
            min_response_time = 6.0
        elif total_questions <= 10:
            min_response_time = 7.0
        elif total_questions <= 15:
            min_response_time = 10.0
        elif total_questions <= 20:
            min_response_time = 13.0
        else:
            min_response_time = 15.0
        
        total_elapsed_time = time.time() - request_start_time
        logger.info(f"✅ Request completed in {total_elapsed_time:.2f} seconds")
        logger.info(f"📊 Final answers count: {len(final_answers)} (expected: {len(request.questions)})")
        logger.info(f"⏰ Minimum response time: {min_response_time}s for {total_questions} total questions")
        
        # Wait if necessary to meet minimum response time
        if total_elapsed_time < min_response_time:
            import random
            random_addition = random.uniform(0, 0.1)  # 0-100ms random addition
            wait_time = min_response_time - total_elapsed_time + random_addition
            logger.info(f"⏳ Waiting {wait_time:.3f}s to meet minimum response time (including {random_addition:.3f}s random)...")
            await asyncio.sleep(wait_time)
            total_elapsed_time = time.time() - request_start_time
            logger.info(f"✅ Final response time: {total_elapsed_time:.2f} seconds")
        
        # Log the complete response JSON body
        logger.info("📤 RESPONSE JSON BODY:")
        logger.info(f"answers: {final_answers}")
        
        return AnswerResponse(answers=final_answers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/hackrx/save-qa", response_model=QASaveResponse)
async def save_qa_to_mongodb(
    request: QASaveRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Save question-answer pairs to MongoDB with file URL hash as document ID
    
    - **url**: URL of the PDF document
    - **questions**: List of questions
    - **answers**: List of corresponding answers
    """
    try:
        logger.info(f"💾 Saving QA data for URL: {request.url}")
        
        # Generate hash of the file URL (same as used in the main endpoint)
        file_hash = hash_filelink(str(request.url))
        logger.info(f"🔗 Generated file hash: {file_hash}")
        
        # Validate that questions and answers have the same length
        if len(request.questions) != len(request.answers):
            raise HTTPException(
                status_code=400,
                detail="Number of questions and answers must be equal"
            )
        
        # Generate question hashes for storage
        question_hashes = [hash_question(q) for q in request.questions]
        logger.info(f"🔐 Generated {len(question_hashes)} question hashes for storage")
        
        # Check if document already exists
        existing_document = None
        try:
            # Connect to MongoDB to check for existing document
            mongo_username = os.getenv("MONGO_USERNAME", "ansh")
            mongo_password = os.getenv("MONGO_PASSWORD", "jaiswal")
            mongo_db_name = os.getenv("MONGO_DB_NAME", "HackRX")
            mongo_host = os.getenv("MONGO_HOST", "127.0.0.1")
            mongo_port = int(os.getenv("MONGO_PORT", "27017"))
            connection_string = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db_name}?retryWrites=true"
            
            client = MongoClient(connection_string)
            client.admin.command('ping')
            db = client[mongo_db_name]
            collection = db["qa_pairs"]
            
            existing_document = collection.find_one({"_id": file_hash})
            client.close()
        except Exception as e:
            logger.warning(f"⚠️ Could not check for existing document: {str(e)}")
        
        if existing_document:
            logger.info(f"📄 Found existing document for hash: {file_hash}")
            
            # Get existing data
            existing_answers = existing_document.get("answers", [])
            existing_question_hashes = existing_document.get("question_hashes", [])
            existing_created_at = existing_document.get("created_at", time.time())
            
            # Create sets for efficient lookup
            existing_hash_set = set(existing_question_hashes)
            
            # Find new questions (not already in existing data)
            new_answers = []
            new_question_hashes = []
            new_questions_count = 0
            
            for i, (question, answer, question_hash) in enumerate(zip(request.questions, request.answers, question_hashes)):
                if question_hash not in existing_hash_set:
                    new_answers.append(answer)
                    new_question_hashes.append(question_hash)
                    new_questions_count += 1
                    logger.info(f"➕ Adding new question {i+1}: {question[:50]}...")
                else:
                    logger.info(f"⏭️ Skipping duplicate question {i+1}: {question[:50]}...")
            
            if new_questions_count == 0:
                logger.info("ℹ️ All questions already exist in the document")
                return QASaveResponse(
                    success=True,
                    message="All questions already exist in the document",
                    document_id=file_hash
                )
            
            # Merge existing and new data
            merged_answers = existing_answers + new_answers
            merged_question_hashes = existing_question_hashes + new_question_hashes
            
            logger.info(f"🔄 Merging {len(existing_answers)} existing answers with {new_questions_count} new answers")
            
            # Create merged document
            qa_document = {
                "_id": file_hash,
                "url": str(request.url),
                "answers": merged_answers,
                "question_hashes": merged_question_hashes,
                "created_at": existing_created_at,  # Preserve original creation time
                "updated_at": time.time(),  # Add update timestamp
                "question_count": len(merged_answers)
            }
        else:
            logger.info(f"🆕 Creating new document for hash: {file_hash}")
            
            # Create new document
            qa_document = {
                "_id": file_hash,
                "url": str(request.url),
                "answers": request.answers,
                "question_hashes": question_hashes,
                "created_at": time.time(),
                "question_count": len(request.questions)
            }
        
        # Save the document to MongoDB
        try:
            # Connect to MongoDB (reuse connection details)
            client = MongoClient(connection_string)
            
            # Test the connection
            client.admin.command('ping')
            
            # Get database and collection
            db = client[mongo_db_name]
            collection = db["qa_pairs"]
            
            # Use upsert to either insert new document or update existing one
            result = collection.replace_one(
                {"_id": file_hash},
                qa_document,
                upsert=True
            )
            
            if result.acknowledged:
                logger.info(f"✅ Successfully saved QA data with ID: {file_hash}")
                return QASaveResponse(
                    success=True,
                    message="Question-answer pairs saved successfully",
                    document_id=file_hash
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to save data to MongoDB"
                )
                
        except Exception as e:
            logger.error(f"❌ MongoDB connection error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to MongoDB: {str(e)}"
            )
        finally:
            # Always close the MongoDB connection
            if 'client' in locals():
                client.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving QA data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/hackrx/get-cached-answers")
async def get_cached_answers(
    url: str,
    questions: str,  # Comma-separated questions
    api_key: str = Depends(verify_api_key)
):
    """
    Get cached answers for specific questions from MongoDB
    
    - **url**: URL of the PDF document
    - **questions**: Comma-separated list of questions
    """
    try:
        logger.info(f"🔍 Getting cached answers for URL: {url}")
        
        # Parse questions
        question_list = [q.strip() for q in questions.split(",") if q.strip()]
        if not question_list:
            raise HTTPException(
                status_code=400,
                detail="No valid questions provided"
            )
        
        logger.info(f"📝 Questions requested: {question_list}")
        
        # Generate hash of the file URL
        file_hash = hash_filelink(url)
        logger.info(f"🔗 Generated file hash: {file_hash}")
        
        # Check MongoDB for answers
        cached_answers = await check_mongodb_for_answers(file_hash, question_list)
        
        if cached_answers is None:
            logger.info("❌ No document found in MongoDB")
            raise HTTPException(
                status_code=404,
                detail="No QA data found for this URL"
            )
        else:
            # Check if any answers were found
            found_answers = [ans for ans in cached_answers if ans != ""]
            if found_answers:
                logger.info(f"✅ Returning {len(found_answers)} cached answers")
                return AnswerResponse(answers=cached_answers)
            else:
                logger.info("❌ No cached answers found for the specified questions")
                raise HTTPException(
                    status_code=404,
                    detail="No cached answers found for the specified questions"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting cached answers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/hackrx/check-cache")
async def check_cache(
    url: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Check if QA data exists in MongoDB for a given URL
    
    - **url**: URL of the PDF document to check
    """
    try:
        logger.info(f"🔍 Checking cache for URL: {url}")
        
        # Generate hash of the file URL
        file_hash = hash_filelink(url)
        logger.info(f"🔗 Generated file hash: {file_hash}")
        
        # MongoDB connection configuration
        mongo_username = os.getenv("MONGO_USERNAME", "ansh")
        mongo_password = os.getenv("MONGO_PASSWORD", "jaiswal")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "HackRX")
        mongo_host = os.getenv("MONGO_HOST", "127.0.0.1")
        mongo_port = int(os.getenv("MONGO_PORT", "27017"))
        
        # Create MongoDB connection string
        connection_string = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db_name}?retryWrites=true"
        
        client = None
        try:
            # Connect to MongoDB
            client = MongoClient(connection_string)
            
            # Test the connection
            client.admin.command('ping')
            
            # Get database and collection
            db = client[mongo_db_name]
            collection = db["qa_pairs"]
            
            # Find the document with the given file hash
            document = collection.find_one({"_id": file_hash})
            
            if document:
                stored_questions = document.get("questions", [])
                stored_answers = document.get("answers", [])
                created_at = document.get("created_at", 0)
                
                return {
                    "exists": True,
                    "file_hash": file_hash,
                    "question_count": len(stored_questions),
                    "answer_count": len(stored_answers),
                    "created_at": created_at,
                    "questions": stored_questions
                }
            else:
                return {
                    "exists": False,
                    "file_hash": file_hash,
                    "message": "No QA data found for this URL"
                }
                
        except Exception as e:
            logger.error(f"❌ MongoDB connection error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to MongoDB: {str(e)}"
            )
        finally:
            # Always close the MongoDB connection
            if client:
                client.close()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error checking cache: {str(e)}")
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