import os
import re
import time
import pickle
import logging
import asyncio
import aiohttp

from groq import AsyncGroq

# Import our existing pipeline components
from app.utils.chunker import chunk_text
from app.utils import sanitize_filename, verify_api_key
from app.core import INSURANCE_SYSTEM_PROMPT , GENERAL_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d', 
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO)


# FastAPI app
app = FastAPI(title="HackRx PDF RAG API", version="1.0.0")


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
            logging.info(f"⚡ Using cached embeddings for {request.documents}")
            with open(pkl_path, "rb") as f:
                process_result = pickle.load(f)
            # Recreate FAISS index and metadata if missing
            index_path = os.path.join(pipeline.vector_store_path, "index.faiss")
            meta_path = os.path.join(pipeline.vector_store_path, "metadata.json")
            import numpy as np
            from app.utils.faiss_store import create_faiss_index, save_metadata
            if not (os.path.exists(index_path) and os.path.exists(meta_path)):
                logging.info("♻️ Recreating FAISS index and metadata from cache...")
                os.makedirs(pipeline.vector_store_path, exist_ok=True)
                create_faiss_index(np.array(process_result["embeddings"]), index_path=index_path)
                save_metadata(process_result["chunks"], meta_path=meta_path)
        else:
            # Download PDF
            logging.info(f"📄 Downloading PDF from: {request.documents}")
            pdf_content = await pipeline.download_pdf(str(request.documents))
            # Process PDF (with cache logic)
            logging.info("🔄 Processing PDF through pipeline...")
            process_result = await pipeline.process_pdf(pdf_content, file_link=str(request.documents))
        # Answer questions using Groq
        logging.info("🤖 Answering questions with Groq...")
        answers = await pipeline.answer_questions(request.questions)
        # Cleanup vector store
        import shutil
        if os.path.exists(pipeline.vector_store_path):
            shutil.rmtree(pipeline.vector_store_path)
        elapsed_time = time.time() - start_time
        logging.info(f"✅ Completed in {elapsed_time:.2f} seconds")
        return AnswerResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
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