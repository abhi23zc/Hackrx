
import os
import logging
import time
import asyncio

from fastapi import status
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.core.template import prompt_template_description

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Async PDF loader
async def pdf_loader(url: str):
    pages = []
    loader = PyPDFLoader(url)
    async for page in loader.alazy_load():
        pages.append(page)
    
    return pages

# Main function to create/load vectorstore
async def load_and_create_vector_store(url: str):
    """
    Loads a PDF document from a URL and either reuses or builds a FAISS vectorstore.
    Returns a retriever object.
    """
    vectorstore_path = "./database/faiss_index"

    if os.path.exists(f"{vectorstore_path}/index.faiss"):
        logging.info("Vector store already exists, loading it.")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        logging.info("Vector store not found. Creating new one from document.")
        pages = await pdf_loader(url)
        if not pages:
            raise ValueError("No pages loaded from the document.")
        
        full_text = "\n\n".join([page.page_content for page in pages])
        documents = [Document(page_content=full_text, metadata={"source": url})]
        # Use CharacterTextSplitter with optimized parameters for better chunk quality
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=2500,
            chunk_overlap=300,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Document split into {len(split_docs)} chunks")

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore.as_retriever(
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )

async def llm_setup(config, url):
    """
    Setup the LLM for question answering.
    
    This function initializes the LLM with the necessary configurations
    for processing questions and generating answers based on the context.
    
    Args:
        config: Configuration dictionary with LLM settings
        url: URL of the document to process
    Returns:
        object: The configured LLM instance.
    """
    llm = ChatGroq(
        model=f"{config.get('MODEL_NAME')}",
        temperature=f"{config.get('TEMPERATURE', 0)}",
        max_tokens=f"{config.get('MAX_TOKENS', 500)}",  # Increased token limit for JSON responses
        max_retries=f"{config.get('MAX_RETRIES', 3)}",
        api_key=f"{config.get('GROQ_KEY')}",
    )
    logging.info(f"LLM initialized with model: {config.get('MODEL_NAME')}, api_key: {config.get('GROQ_KEY')}")

    # Choose template based on whether we need structured JSON output
    prompt_template = prompt_template_description()

    retriever = await load_and_create_vector_store(url=url)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

async def llm_response_generator(config, url, questions):
    """
    Generate answers from the LLM within 30 seconds.
    
    Args:
        config: Configuration dictionary with LLM settings
        url: URL of the document to process
        questions: List of questions to answer
        use_json: Whether to force JSON output format
        
    Returns:
        Tuple of (response dict, status code)
    """
    try:
        start = time.time()
        qa_chain = await llm_setup(config, url)

        answers = []
        for question in questions:
            elapsed = time.time() - start
            if elapsed > 28:  # leave margin for safety
                logging.warning("Time limit reached, skipping remaining questions.")
                break

            try:
                answer = await qa_chain.arun(question)
                answers.append(answer)
            
            except Exception as e:
                logging.error(f"Error answering: {question} | {e}")
                answers.append("Error processing question.")

        return {"answers": answers}, status.HTTP_200_OK
    
    except Exception as e:
        logging.error(f"Error in llm_response_generator: {e}")
        return {"answers": []}, status.HTTP_500_INTERNAL_SERVER_ERROR
