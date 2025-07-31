import os
import logging
import uvicorn
import json
import hashlib

# Import our existing pipeline components
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from app.utils.util import verify_api_key
from app.core.models import llm_response_generator
from app.schema.schema import QuestionRequest, AnswerResponse

logging.basicConfig(format='%(asctime)s - %(levelname)s - Line: %(lineno)d - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load config.json at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    import json
    config_path = os.path.join(os.path.dirname(__file__), "app", "config", "config.json")
    with open(config_path, "r") as f:
        app.state.config = json.load(f)
    logging.info("Config loaded successfully.")
    yield
    logging.info("Shutting down FastAPI app...")


# FastAPI app
app = FastAPI(title="HackRx PDF RAG API", version="1.0.0", lifespan=lifespan)

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def process_questions(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    try:
        url = str(request.documents)
        config = app.state.config
        questions = request.questions

        logging.info(f"Received {len(questions)} questions for processing. Documents URL: {url}")

        # Create cache directory if not exists
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Create a cache key from url and questions
        cache_key = hashlib.sha256((url + json.dumps(questions, sort_keys=True)).encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")

        # Try to load from cache
        if os.path.exists(cache_file):
            logging.info(f"Cache hit for key: {cache_key}")
            with open(cache_file, "r") as f:
                cached_response = json.load(f)
            return cached_response

        # Otherwise, call LLM and cache response
        response, status_code = await llm_response_generator(config=config, url=url, questions=questions)
        if status_code != status.HTTP_200_OK:
            raise HTTPException(status_code=status_code, detail="Error processing questions")
        
        else:
            with open(cache_file, "w") as f:
                json.dump(response, f)
            logging.info(f"Cache saved for key: {cache_key}")
            return response

    except Exception as e:
        logging.error(f"Error processing questions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {"status": "Healthy", "message": "API is running smoothly.", "code": status.HTTP_200_OK}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )