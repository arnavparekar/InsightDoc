import os
import time
import logging
import asyncio
import uuid
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# --- Configuration ---
os.environ["GROQ_API_KEY"] = "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"
# Silences the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="An LLM-powered system for contextual document analysis.",
    version="1.0.0",
)

# --- Security ---
security = HTTPBearer()
EXPECTED_TOKEN = "2a689d9dd2262c6f52836c0e0dbaf8f6a06df36e67bebc4a618d0afabcbbd50f"

async def verify_token(credentials: str = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models for API ---
class HackRxRunRequest(BaseModel):
    documents: str = Field(..., example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D")
    questions: List[str] = Field(..., example=[
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
    ])

class HackRxRunResponse(BaseModel):
    answers: List[str]

# --- Core Logic ---
async def process_single_question(question: str, retriever, rag_chain: Runnable):
    """Asynchronously processes a single question against the RAG chain."""
    return await rag_chain.ainvoke(question)

async def process_document_and_query(doc_url: str, questions: List[str]):
    """
    Processes a document from a URL, creates a vector store, and answers questions in parallel.
    """
    try:
        # 1. Load Document
        response = requests.get(doc_url)
        response.raise_for_status()
        temp_pdf_path = f"temp_document_{uuid.uuid4()}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # 2. Chunk Document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # 4. Initialize LLM
        llm = ChatGroq(model_name="llama3-8b-8192")

        # 5. Define RAG Chain
        prompt_template = """Based *only* on the following context, provide a clear and concise answer to the question. Do not mention the context or the documents in your answer. Synthesize the information into a final, clean response.provide a clear and concise answer to the question. IMPORTANT: Your answer must be a single, complete sentence. For example, if the answer is a number even then respond with complete sentence.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 6. Process Questions in Parallel
        tasks = [rag_chain.ainvoke(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        
        os.remove(temp_pdf_path)

        return [ans.replace('\n', ' ').replace('*', '').strip() for ans in answers]

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxRunResponse)
async def hackrx_run(request: HackRxRunRequest = Body(...), _ = Depends(verify_token)):
    """
    Run the intelligent query-retrieval system.
    """
    logger.info("--- New Request Received ---")
    logger.info(f"Document URL: {request.documents}")
    logger.info("Questions:")
    for i, question in enumerate(request.questions, 1):
        logger.info(f"  {i}. {question}")

    start_time = time.time()

    answers = await process_document_and_query(request.documents, request.questions)

    end_time = time.time()
    time_taken = end_time - start_time

    cleaned_url = urlsplit(request.documents)._replace(query=None, fragment=None).geturl()
    logger.info(f"Request processed successfully in {time_taken:.2f} seconds.")
    logger.info(f"Document URL: {cleaned_url}")

    return HackRxRunResponse(answers=answers)

# --- Main Entry Point for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)