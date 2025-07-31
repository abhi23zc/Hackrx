import os
import logging
from app.utils.pdf_extractor import extract_pdf_content
from app.utils.prompt_builder import build_prompt_without_sources
from app.utils.faiss_store import create_faiss_index, save_metadata
from retriever_reranker import retrieve_top_k, rerank_chunks
from app.utils.embedder import model, generate_embeddings, generate_openai_embeddings
from app.core import INSURANCE_SYSTEM_PROMPT , GENERAL_SYSTEM_PROMPT

class PDFRAGPipeline:
    def __init__(self):
        self.setup_groq()
        self.vector_store_path = "vector_store"
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
            self.model_name = "llama-3.3-70b-versatile"  # Most efficient model gemma qwen mistral
            logging.info("✅ Groq API configured successfully")
        except Exception as e:
            logging.error(f"❌ Failed to configure Groq: {e}")
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
            # Save PDF temporarily
            temp_pdf_path = "temp_document.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)
            # Step 1: Extract PDF content
            logging.info("🔍 Extracting PDF content...")
            pdf_data = extract_pdf_content(temp_pdf_path)
            logging.info(f"✅ Extracted {len(pdf_data['pages'])} pages")
            # Step 2: Chunk the text (reverted to RecursiveCharacterTextSplitter)
            logging.info("✂️ Chunking text...")
            chunks = chunk_text(pdf_data["pages"])
            logging.info(f"✅ Created {len(chunks)} chunks")
            # Step 3: Generate embeddings (batch processing)
            logging.info("🧠 Generating embeddings with HuggingFace model...")
            texts, embeddings = generate_embeddings(chunks)
            logging.info(f"✅ Generated embeddings: {getattr(embeddings, 'shape', type(embeddings))}")
            # Step 4: Store in FAISS
            logging.info("💾 Storing in vector database...")
            os.makedirs(self.vector_store_path, exist_ok=True)
            create_faiss_index(
                embeddings,
                index_path=os.path.join(self.vector_store_path, "index.faiss")
            )
            save_metadata(
                chunks,
                meta_path=os.path.join(self.vector_store_path, "metadata.json")
            )
            logging.info("✅ Vector store created successfully")
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
            logging.info(f"🔍 Processing question: {question}")

            # Step 1: Retrieve relevant chunks
            retrieved = retrieve_top_k(question, k=5)
            logging.info(f"✅ Retrieved {len(retrieved)} relevant chunks")
            if not retrieved:
                logging.error("No relevant chunks retrieved for question.")
                return "Information not available in the provided document."

            # Step 2: Rerank chunks
            reranked = rerank_chunks(question, retrieved, top_n=3)
            logging.info(f"✅ Reranked to top {len(reranked)} chunks")
            if not reranked:
                logging.error("No chunks after reranking.")
                return "Information not available in the provided document."

            # Log the full reranked chunks being sent with the prompt
            logging.info("📦 Chunks sent with prompt:")
            for i, chunk in enumerate(reranked):
                logging.info(f"Chunk {i}: {chunk}")

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
            logging.info("✅ Generated answer for question")

            return answer

        except Exception as e:
            logging.error(f"Error processing question: {repr(e)}", exc_info=True)
            return f"I encountered an error answering this question: {str(e)}"