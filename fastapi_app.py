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
INSURANCE_SYSTEM_PROMPT = (
    "You are a specialized AI assistant for health insurance policy analysis. Provide precise, factual answers based on the policy document.\n\n"
    "CRITICAL RULES:\n"
    "- Each context chunk will have a similarity score in the format [Score: X.XXXX].\n"
    "- If the highest similarity score is < 0.45, respond with: \"Information not available in the provided document.\"\n"
    "- Answer exactly what is asked with the most important details only\n"
    "- Include specific numbers, time periods, and key conditions\n"
    "- Keep answers to 1-2 sentences maximum\n"
    "- Use clear, professional language\n"
    "- Focus on the core information requested\n"
    "- If information is not in the context, respond with: \"Information not available in the provided document.\"\n\n"
    "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources. Provide a focused answer with only the essential policy details that directly answer the question."
)


GENERAL_SYSTEM_PROMPT = (
    "You are a HUMAN subject matter expert based strictly on the context of the provided document.\n"
    "These documents may include anything.\n\n"
    "CRITICAL RULES:\n"
    "- Each context chunk will have a similarity score in the format [Score: X.XXXX].\n"
    "- Higher similarity scores indicate more relevant information.\n"
    "- If after using the chunks and all analysis you can't find relevant information, use your own capabilities to answer, but first write: \"Couldn't find relevant information in the document but here's an answer.\" Then provide your answer.\n"
    "- Use clear, professional language.\n"
    "- Focus on the core information requested.\n"
    "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources. Provide a focused answer with only essential details."
)

# QWEN-style system prompt for concise, direct, factual answers (no thinking steps, no context explanation)
QWEN_SYSTEM_PROMPT = (
    "You are a highly knowledgeable assistant. Answer ONLY with the direct, factual answer to the user's question, based strictly on the provided document context.\n\n"
    "RULES:\n"
    "- Each context chunk will have a similarity score in the format [Score: X.XXXX].\n"
    "- If the highest similarity score is < 0.45, reply exactly: 'Information not available in the provided document.'\n"
    "- Do NOT include any reasoning, thinking steps, or explanations.\n"
    "- Do NOT mention the context, pages, or your process.\n"
    "- Do NOT use phrases like 'Based on the document' or 'Looking at the context'.\n"
    "- If the answer is not found in the document, reply exactly: 'Information not available in the provided document.'\n"
    "- Use clear, concise, and professional language.\n"
    "- Include specific numbers, time periods, and key conditions if present.\n"
    "- Keep answers to 1-2 sentences, as in a summary.\n"
    "- Do NOT include any <think> or meta-cognitive steps.\n"
    "- Respond ONLY with the answer text.\n\n"
    "EXAMPLES:\n"
    "Q: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\n"
    "A: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.\n\n"
    "Q: Does this policy cover maternity expenses, and what are the conditions?\n"
    "A: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.\n"
)

# Remove token limit and estimation

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
            # Use hardcoded OpenRouter API key
            openrouter_api_key = "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"
            self.openrouter_llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct",
                openai_api_key=openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
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
            # Not cached: process as usual
            temp_pdf_path = "temp_document.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)
            logger.info("🔍 Extracting PDF content...")
            pdf_data = await extract_pdf_content(temp_pdf_path)
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
            os.remove(temp_pdf_path)
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

    async def answer_questions(self, questions: List[str], llm_provider: str = "groq", file_hash: str = None) -> List[str]:
        """Answer questions using the selected LLM provider with a 1s gap between each call."""
        try:
            answers = []
            tasks = []
            for i, question in enumerate(questions):
                # Stagger each call by 1 second
                async def delayed_call(q=question, delay=i):
                    await asyncio.sleep(delay)
                    return await self._process_single_question(q, llm_provider=llm_provider, file_hash=file_hash)
                tasks.append(delayed_call())
            answers = await asyncio.gather(*tasks)
            return answers
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error answering questions: {str(e)}"
            )

    async def _process_single_question(self, question: str, llm_provider: str = "groq", file_hash: str = None) -> str:
        try:
            logger.info(f"🔍 Processing question: {question} (LLM: {llm_provider})")
            # Step 1: Retrieve relevant chunks using hash-based paths
            index_path = os.path.join(self.embeddings_dir, f"{file_hash}_index.faiss")
            meta_path = os.path.join(self.embeddings_dir, f"{file_hash}_metadata.json")
            retrieved = retrieve_top_k(question, k=5, index_path=index_path, meta_path=meta_path)
            logger.info(f"✅ Retrieved {len(retrieved)} relevant chunks")
            if not retrieved:
                logger.error("No relevant chunks retrieved for question.")
                return "Information not available in the provided document."
            # Step 2: Rerank chunks
            reranked = rerank_chunks(question, retrieved, top_n=3)
            logger.info(f"✅ Reranked to top {len(reranked)} chunks")
            if not reranked:
                logger.error("No chunks after reranking.")
                return "Information not available in the provided document."
            logger.info("📦 Chunks sent with prompt:")
            for i, chunk in enumerate(reranked):
                logger.info(f"Chunk {i}: {chunk}")
            # Step 3: Build optimized prompt with similarity scores
            prompt = build_prompt_without_sources(question, reranked)
            system_prompt = GENERAL_SYSTEM_PROMPT
            
            # Log the complete prompt being sent to LLM
            logger.info("🤖 FINAL PROMPT BEING SENT TO LLM:")
            logger.info("=" * 80)
            logger.info(f"System Prompt: {system_prompt}")
            logger.info("-" * 80)
            logger.info(f"User Prompt: {prompt}")
            logger.info("=" * 80)
            # Step 4: Get answer from the selected LLM
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if llm_provider == "openai":
                        if not self.openai_api_key:
                            logger.error("OPENAI_API_KEY not set. Cannot use OpenAI LLM.")
                            return "OpenAI API key not configured."
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
                        # Log the full response object to see headers and metadata
                        logger.info(f"🔍 Full LLM Response Object: {response}")
                        logger.info(f"🔍 Response Type: {type(response)}")
                        logger.info(f"🔍 Response Attributes: {dir(response)}")
                        if hasattr(response, 'response_metadata'):
                            logger.info(f"🔍 Response Metadata: {response.response_metadata}")
                        if hasattr(response, 'llm_output'):
                            logger.info(f"🔍 LLM Output: {response.llm_output}")
                        answer = response.content.strip()
                        logger.info(f"🔍 Extracted Answer: {answer}")
                    logger.info("✅ Generated answer for question")
                    return answer
                except Exception as e:
                    # Check for 429 error
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited by LLM API. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            return "Rate limit exceeded. Please try again later."
        except Exception as e:
            logger.error(f"Error processing question: {repr(e)}", exc_info=True)
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
        logger.info(f"📄 Downloading PDF from: {request.documents}")
        file_hash = hash_filelink(str(request.documents))
        pkl_path = os.path.join(pipeline.embeddings_dir, f"{file_hash}.pkl")
        # Optimization: Check if hash is already processed before downloading
        if file_hash in pipeline.processed_hashes and os.path.exists(pkl_path):
            logger.info(f"⚡ Cache hit for document hash: {file_hash}. Using cached embeddings, skipping download and processing.")
            with open(pkl_path, "rb") as f:
                process_result = pickle.load(f)
        else:
            pdf_content = await pipeline.download_pdf(str(request.documents))
            logger.info("🔄 Processing PDF through pipeline...")
            process_result = await pipeline.process_pdf(pdf_content, file_link=str(request.documents))
        # Answer questions using Groq
        logger.info("🤖 Answering questions with Groq...")
        answers = await pipeline.answer_questions(request.questions, llm_provider="groq", file_hash=file_hash)
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