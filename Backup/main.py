import os
import sys
import json
import time
from pathlib import Path

# Import all our pipeline components
from pdf_extractor import extract_pdf_content
from chunker import chunk_text
from embedder import generate_embeddings
from faiss_store import create_faiss_index, save_metadata
from retriever_reranker import retrieve_top_k, rerank_chunks
from prompt_builder import build_prompt_without_sources

import google.generativeai as genai

class PDFRAGPipeline:
    def __init__(self, api_key=None):
        """
        Initialize the PDF RAG Pipeline
        
        Args:
            api_key: Gemini API key (optional, will use hardcoded if not provided)
        """
        self.api_key = api_key or "AIzaSyAb2K0HUEY2b7lqcwE6qUrcxByxUN3D6ds"
        self.setup_gemini()
        self.vector_store_path = "vector_store"
        
    def setup_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-pro")
            print("✅ Gemini API configured successfully")
        except Exception as e:
            print(f"❌ Failed to configure Gemini: {e}")
            sys.exit(1)
    
    def process_pdf(self, pdf_path):
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict: Processing results with metadata
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
        
        print(f"📄 Processing PDF: {pdf_path}")
        
        try:
            # Step 1: Extract PDF content
            print("🔍 Extracting PDF content...")
            pdf_data = extract_pdf_content(pdf_path)
            print(f"✅ Extracted {len(pdf_data['pages'])} pages")
            
            # Step 2: Chunk the text
            print("✂️ Chunking text...")
            chunks = chunk_text(pdf_data["pages"])
            print(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("🧠 Generating embeddings...")
            texts, embeddings = generate_embeddings(chunks)
            print(f"✅ Generated embeddings: {embeddings.shape}")
            
            # Step 4: Store in FAISS
            print("💾 Storing in vector database...")
            os.makedirs(self.vector_store_path, exist_ok=True)
            create_faiss_index(embeddings, 
                             index_path=os.path.join(self.vector_store_path, "index.faiss"))
            save_metadata(chunks, 
                        meta_path=os.path.join(self.vector_store_path, "metadata.json"))
            print("✅ Vector store created successfully")
            
            return {
                "success": True,
                "pages": len(pdf_data['pages']),
                "chunks": len(chunks),
                "embeddings_shape": embeddings.shape,
                "vector_store_path": self.vector_store_path
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def query_document(self, question, top_k=10, top_n=5):
        """
        Query the processed document
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve initially
            top_n: Number of chunks to use for final answer
            
        Returns:
            dict: Query results with answer and metadata
        """
        if not os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            return {"error": "No document processed. Please upload and process a PDF first."}
        
        try:
            print(f"🔍 Querying: {question}")
            
            # Step 1: Retrieve relevant chunks
            retrieved = retrieve_top_k(question, k=top_k)
            print(f"✅ Retrieved {len(retrieved)} relevant chunks")
            
            # Step 2: Rerank chunks
            reranked = rerank_chunks(question, retrieved, top_n=top_n)
            print(f"✅ Reranked to top {len(reranked)} chunks")
            
            # Step 3: Build prompt and generate answer
            prompt = build_prompt_without_sources(question, reranked)
            
            # Step 4: Get answer from Gemini
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources_used": len(reranked),
                "source_pages": list(set([chunk['page'] for chunk in reranked]))
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Interactive CLI interface"""
    pipeline = PDFRAGPipeline()
    
    print("🚀 PDF RAG Pipeline Started")
    print("=" * 50)
    
    while True:
        print("\n📋 Menu:")
        print("1. Upload and Process PDF")
        print("2. Ask Question")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            pdf_path = input("📄 Enter PDF file path: ").strip().strip('"')
            if os.path.exists(pdf_path):
                result = pipeline.process_pdf(pdf_path)
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("\n✅ PDF processed successfully!")
                    print(f"📊 Pages: {result['pages']}")
                    print(f"📊 Chunks: {result['chunks']}")
                    print(f"📊 Embeddings: {result['embeddings_shape']}")
            else:
                print("❌ File not found. Please check the path.")
                
        elif choice == "2":
            question = input("❓ Ask your question: ").strip()
            if question:
                result = pipeline.query_document(question)
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n🤖 Answer: {result['answer']}")
                    print(f"📖 Sources from pages: {result['source_pages']}")
            else:
                print("❌ Please enter a question.")
                
        elif choice == "3":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
